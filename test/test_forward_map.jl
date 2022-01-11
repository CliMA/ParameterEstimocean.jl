using Test
using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

using OceanTurbulenceParameterEstimation.Observations: FieldTimeSeriesCollector, initialize_simulation!, observation_times
using OceanTurbulenceParameterEstimation.InverseProblems: transpose_model_output

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
        grid = RectilinearGrid(size=size, z=(-128, 0), topology=(Flat, Flat, Bounded))
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
    observations = SyntheticObservations(data_path, field_names=(:u, :b), normalize=ZScore)

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

        ensemble_size = 1
        column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(ensemble_size, 1), Hz=1)
        ensemble_grid = RectilinearGrid(size=column_ensemble_size, z = (-128, 0), topology = (Flat, Flat, Bounded))
        closure_ensemble = [default_closure() for i = 1:ensemble_grid.Nx, j = 1:ensemble_grid.Ny]

        N² = 1e-5
        ensemble_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4))
        ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(1e-7), bottom = GradientBoundaryCondition(N²))

        ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                                    tracers = :b,
                                                    buoyancy = BuoyancyTracer(),
                                                    boundary_conditions = (; u=ensemble_u_bcs, b=ensemble_b_bcs),
                                                    closure = closure_ensemble)

        set!(ensemble_model, b = (x, y, z) -> N² * z)

        ensemble_simulation = Simulation(ensemble_model; Δt=20.0, stop_iteration=3*60*10)

        calibration = InverseProblem(observations, ensemble_simulation, free_parameters)
        optimal_parameters = [getproperty(closure, p) for p in keys(priors)]
        
        x₁ = forward_map(calibration,  [optimal_parameters for _ in 1:1])
        x₂ = forward_map(calibration,  [optimal_parameters for _ in 1:1])

        y = observation_map(calibration)
        
        @test x₁[:, 1:1] == y
        @test x₂[:, 1:1] == y
    end

    @testset "Two-member (2x1) transposition of model output" begin
        ensemble_size = 2
        column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(ensemble_size, 1), Hz=1)
        test_simulation = build_simulation(column_ensemble_size)
        collected_fields = (u = test_simulation.model.velocities.u, b = test_simulation.model.tracers.b)
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))
        initialize_simulation!(test_simulation, observations, time_series_collector)
        run!(test_simulation)
        output = transpose_model_output(time_series_collector, observations)

        truth_b = observations.field_time_serieses.b
        truth_u = observations.field_time_serieses.u

        test_b = output[1].field_time_serieses.b
        test_u = output[1].field_time_serieses.u

        for n in 1:ensemble_size
            @test test_b[n, :, :, :] == truth_b[1, :, : ,:]
            @test test_u[n, :, :, :] == truth_u[1, :, : ,:]
        end
    end

    @testset "Six-member (2x3) transposition of model output" begin
        ensemble_size = 2
        batch_size = 3
        observations_batch = [observations, observations, observations]
        column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(ensemble_size, batch_size), Hz=1)
        test_simulation = build_simulation(column_ensemble_size)
        collected_fields = (u = test_simulation.model.velocities.u, b = test_simulation.model.tracers.b)
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))
        initialize_simulation!(test_simulation, observations_batch, time_series_collector)
        run!(test_simulation)
        output = transpose_model_output(time_series_collector, observations_batch)

        for j in 1:batch_size
            truth_b = observations_batch[j].field_time_serieses.b
            truth_u = observations_batch[j].field_time_serieses.u
    
            test_b = output[j].field_time_serieses.b
            test_u = output[j].field_time_serieses.u
        
            for n in 1:ensemble_size
                @test test_b[n, :, :, :] == truth_b[1, :, : ,:]
                @test test_u[n, :, :, :] == truth_u[1, :, : ,:]
            end
        end
    end
end
