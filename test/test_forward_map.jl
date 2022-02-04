using Test
using LinearAlgebra
using Distributions
using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: halo_size
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

using OceanTurbulenceParameterEstimation.Observations: FieldTimeSeriesCollector, initialize_simulation!, observation_times
using OceanTurbulenceParameterEstimation.InverseProblems: transpose_model_output, forward_run!, drop_y_dimension

@testset "Unit tests for forward_map" begin
    # Test drop_y_dimension
    column_ensemble_size = ColumnEnsembleSize(Nz=8, ensemble=(2, 3))
    column_ensemble_halo_size = ColumnEnsembleSize(Nz=1, Hz=5)
    ensemble_grid = RectilinearGrid(size = column_ensemble_size,
                                    halo = column_ensemble_halo_size,
                                    z = (-128, 0),
                                    topology = (Flat, Flat, Bounded))

    dropped_y_grid = drop_y_dimension(ensemble_grid)

    @test size(ensemble_grid) = (2, 3, 8)
    @test halo_size(ensemble_grid) = (0, 0, 5)
    @test size(dropped_y_grid) = (2, 1, 8)
    @test halo_size(dropped_y_grid) = (0, 0, 5)
end

@testset "Forward map tests" begin

    # Generate synthetic observations
    Nz = 8
    N² = 1e-5
    stop_iteration = 3*60*10
    experiment_name = "forward_map_test"

    default_closure() = ConvectiveAdjustmentVerticalDiffusivity(; convective_κz=1.0,
                                                                  background_κz=1e-5,
                                                                  convective_νz=0.9,
                                                                  background_νz=1e-4)

    model_kwargs = (
        tracers = :b,
        buoyancy = BuoyancyTracer(),
        coriolis = FPlane(f=1e-4),
        boundary_conditions = (u = FieldBoundaryConditions(top = FluxBoundaryCondition(-1e-4)),
                               b = FieldBoundaryConditions(top = FluxBoundaryCondition(1e-7), bottom = GradientBoundaryCondition(N²)))
    )

    function build_simulation(size=Nz)
        grid = RectilinearGrid(size=size, z=(-128, 0), topology=(Flat, Flat, Bounded))
        model = HydrostaticFreeSurfaceModel(; grid,
                                              closure = default_closure(),
                                              model_kwargs...)
                                        
        set!(model, b = (x, y, z) -> N² * z)

        simulation = Simulation(model; Δt=20.0, stop_iteration)

        return simulation
    end

    # Make observations
    truth_simulation = build_simulation()
    model = truth_simulation.model
    truth_simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                                schedule = IterationInterval(round(Int, stop_iteration / 10)),
                                                                prefix = experiment_name,
                                                                array_type = Array{Float64},
                                                                field_slicer = nothing,
                                                                force = true)

    run!(truth_simulation)

    data_path = experiment_name * ".jld2"
    observations = SyntheticObservations(data_path, field_names=(:u, :b), normalization=(u=RescaledZScore(0.1), b=ZScore()))
    
    #####
    ##### Make model data
    #####

    @testset "Single-member forward map" begin
        # First test transpose_model_output
        test_simulation = build_simulation()
        collected_fields = (u = test_simulation.model.velocities.u, b = test_simulation.model.tracers.b)
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))

        # Test initialize_simulation!
        @info "  Testing initialize_simulation!..."
        random_initial_condition(x, y, z) = rand()

        for field in fields(test_simulation.model)
            set!(field, random_initial_condition)
        end

        test_u = test_simulation.model.velocities.u
        test_v = test_simulation.model.velocities.v
        test_b = test_simulation.model.tracers.b
        @test !all(test_u .== 0)
        @test !all(test_v .== 0)
        @test !all(test_b .== 0)

        test_simulation.stop_iteration = Inf
        initialize_simulation!(test_simulation, observations, time_series_collector)

        @test all(test_v .== 0)

        run!(test_simulation)

        @info "  Testing transpose_model_output..."
        output = transpose_model_output(time_series_collector, observations)[1]

        test_b = output.field_time_serieses.b
        truth_b = observations.field_time_serieses.b
        test_u = output.field_time_serieses.u
        truth_u = observations.field_time_serieses.u

        @test interior(test_b) == interior(truth_b)
        @test interior(test_u) == interior(truth_u)

        @info "  Testing forward_map and output_map..."
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

        ensemble_model = HydrostaticFreeSurfaceModel(; grid = ensemble_grid,
                                                       closure = closure_ensemble,
                                                       model_kwargs...)

        ensemble_simulation = Simulation(ensemble_model; Δt=20.0)

        calibration = InverseProblem(observations, ensemble_simulation, free_parameters)
        optimal_parameters = NamedTuple(name => getproperty(closure, name) for name in keys(priors))

        forward_run!(calibration, optimal_parameters)
        truth_u = observations.field_time_serieses.u
        test_u = calibration.time_series_collector.field_time_serieses.u
        @test interior(test_b) == interior(truth_b)
        @test interior(test_u) == interior(truth_u)

        forward_run!(calibration, optimal_parameters)
        truth_u = observations.field_time_serieses.u
        test_u = calibration.time_series_collector.field_time_serieses.u
        @test interior(test_b) == interior(truth_b)
        @test interior(test_u) == interior(truth_u)
        
        x₁ = forward_map(calibration, optimal_parameters)
        x₂ = forward_map(calibration, optimal_parameters)

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
