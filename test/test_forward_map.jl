using Test
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.Observations: FieldTimeSeriesCollector, initialize_simulation!, observation_times
using OceanTurbulenceParameterEstimation.InverseProblems:  transpose_model_output

@testset "Forward map tests" begin

    Nz = 8
    experiment_name = "forward_map_test"

    # Generate synthetic observations

    default_closure() = ConvectiveAdjustmentVerticalDiffusivity(; convective_κz=1.0,
                                                                  background_κz=1e-5,
                                                                  convective_νz=0.9,
                                                                  background_νz=1e-4)

    function build_simulation(size=Nz)
        N² = 1e-5
        grid = RegularRectilinearGrid(size=size, z=(-128, 0), topology=(Flat, Flat, Bounded))
        u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4))
        b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(1e-7), bottom = GradientBoundaryCondition(N²))
        model = HydrostaticFreeSurfaceModel(grid = grid,
                                            tracers = :b,
                                            buoyancy = BuoyancyTracer(),
                                            boundary_conditions = (; u=u_bcs, b=b_bcs),
                                            closure = default_closure())
                                        
        set!(model, b = (x, y, z) -> N² * z)

        simulation = Simulation(model; Δt=20.0, stop_iteration=3*60*10)

        return simulation
    end

    # Make observations

    truth_simulation = build_simulation()
    model = truth_simulation.model
    stop_iteration = truth_simulation.stop_iteration
    truth_simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                                schedule = IterationInterval(round(Int, stop_iteration / 10)),
                                                                prefix = experiment_name,
                                                                array_type = Array{Float64},
                                                                field_slicer = nothing,
                                                                force = true)

    run!(truth_simulation)

    data_path = experiment_name * ".jld2"
    observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :b), normalize=ZScore)

    #####
    ##### Make model data
    #####

    @testset "Single-member forward map" begin
        # First test transpose_model_output
        test_simulation = build_simulation()
        collected_fields = (u = test_simulation.model.velocities.u, b = test_simulation.model.tracers.b)

        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))

        initialize_simulation!(test_simulation, observations, time_series_collector)
        run!(test_simulation)

        output = transpose_model_output(time_series_collector, observations)[1]

        test_b = output.field_time_serieses.b
        truth_b = observations.field_time_serieses.b
        test_u = output.field_time_serieses.u
        truth_u = observations.field_time_serieses.u

        @test interior(test_b) == interior(truth_b)
        @test interior(test_u) == interior(truth_u)

        # Now test forward_map and observation_map
        closure = default_closure()

        priors = (
            convective_κz = Normal(closure.convective_κz, 0.05),
            background_κz = Normal(closure.background_κz, 1e-5),
            convective_νz = Normal(closure.convective_νz, 0.01),
            background_νz = Normal(closure.background_νz, 1e-5),
        )
        
        free_parameters = FreeParameters(priors)
        calibration = InverseProblem(observations, ensemble_simulation, free_parameters)
        optimal_parameters = [getproperty(closure, p) for p in keys(priors)]

        x = forward_map(calibration, optimal_parameters)[1:1, :]
        y = observation_map(calibration)

        @test x == y
    end

    @testset "Two-member transposition of model output" begin
        ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(2, 1), Hz=1)
        test_simulation = build_simulation(ensemble_size)
        collected_fields = (u = test_simulation.model.velocities.u, b = test_simulation.model.tracers.b)
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))
        initialize_simulation!(test_simulation, observations, time_series_collector)
        run!(test_simulation)
        output = transpose_model_output(time_series_collector, observations)

        truth_b = observations.field_time_serieses.b
        truth_u = observations.field_time_serieses.u

        test_b1 = output[1].field_time_serieses.b
        test_u1 = output[1].field_time_serieses.u
        test_b2 = output[2].field_time_serieses.b
        test_u2 = output[2].field_time_serieses.u

        @test interior(test_b1) == interior(truth_b)
        @test interior(test_u1) == interior(truth_u)
        @test interior(test_b2) == interior(truth_b)
        @test interior(test_u2) == interior(truth_u)
    end

    @testset "Four-member transposition of model output" begin
        ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(2, 2), Hz=1)
        test_simulation = build_simulation(ensemble_size)
        collected_fields = (u = test_simulation.model.velocities.u, b = test_simulation.model.tracers.b)
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))
        initialize_simulation!(test_simulation, observations, time_series_collector)
        run!(test_simulation)
        output = transpose_model_output(time_series_collector, observations)

        truth_b = observations.field_time_serieses.b
        truth_u = observations.field_time_serieses.u

        test_b1 = output[1].field_time_serieses.b
        test_u1 = output[1].field_time_serieses.u
        test_b2 = output[2].field_time_serieses.b
        test_u2 = output[2].field_time_serieses.u
        test_b3 = output[3].field_time_serieses.b
        test_u3 = output[3].field_time_serieses.u
        test_b4 = output[4].field_time_serieses.b
        test_u4 = output[4].field_time_serieses.u

        @test interior(test_b1) == interior(truth_b)
        @test interior(test_u1) == interior(truth_u)
        @test interior(test_b2) == interior(truth_b)
        @test interior(test_u2) == interior(truth_u)
        @test interior(test_b3) == interior(truth_b)
        @test interior(test_u3) == interior(truth_u)
        @test interior(test_b4) == interior(truth_b)
        @test interior(test_u4) == interior(truth_u)
    end
end
