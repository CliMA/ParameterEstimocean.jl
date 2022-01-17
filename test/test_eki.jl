using Test
using JLD2
using Statistics
using LinearAlgebra
using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.EnsembleKalmanInversions: iterate!, FullEnsembleDistribution, NaNResampler, resample!
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize

data_path = "convective_adjustment_test.jld2"
Nensemble = 3

@testset "EnsembleKalmanInversions tests" begin
    #####
    ##### Build InverseProblem
    #####

    observation = SyntheticObservations(data_path, field_names=(:u, :v, :b))

    Nz = observation.grid.Nz
    Hz = observation.grid.Hz
    Lz = observation.grid.Lz
    Δt = observation.metadata.parameters.Δt

    Qᵘ, Qᵇ, N², f = [zeros(Nensemble, 1) for i = 1:4]
    Qᵘ[:, 1] .= observation.metadata.parameters.Qᵘ
    Qᵇ[:, 1] .= observation.metadata.parameters.Qᵇ
    N²[:, 1] .= observation.metadata.parameters.N²
    f[:, 1] .= observation.metadata.coriolis.f

    file = jldopen(observation.path)
    closure = file["serialized/closure"]
    close(file)

    column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(Nensemble, 1), Hz=Hz)
    ensemble_grid = RectilinearGrid(size = column_ensemble_size, topology = (Flat, Flat, Bounded), z = (-Lz, 0))

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))

    ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                                 tracers = (:b,),
                                                 buoyancy = BuoyancyTracer(),
                                                 boundary_conditions = (; u=u_bcs, b=b_bcs),
                                                 coriolis = FPlane.(f),
                                                 closure = [deepcopy(closure) for i = 1:Nensemble, j=1:1])

    ensemble_simulation = Simulation(ensemble_model; Δt=Δt, stop_time=observation.times[end])

    lb, ub = [0.9, 1.1]
    priors = (
        convective_κz = ConstrainedNormal(0.0, 1.0, lb, ub),
        background_κz = ConstrainedNormal(0.0, 1.0, 5e-5, 2e-4)
    )
    
    Nparams = length(priors)
    
    free_parameters = FreeParameters(priors)

    calibration = InverseProblem(observation, ensemble_simulation, free_parameters)

    #####
    ##### Test EKI
    #####

    eki = EnsembleKalmanInversion(calibration; noise_covariance = 0.01)

    iterations = 5
    iterate!(eki; iterations = iterations, show_progress = false)

    @test length(eki.iteration_summaries) == iterations + 1
    @test eki.iteration == iterations

    θθ_std_arr = sqrt.(hcat(diag.(getproperty.(eki.iteration_summaries, :ensemble_cov))...))
    @test size(θθ_std_arr) == (Nparams, iterations + 1)

    # Test that parameters stay within bounds
    parameters = eki.iteration_summaries[0].parameters
    convective_κzs = getproperty.(parameters, :convective_κz)
    @test all(convective_κzs .> lb)
    @test all(convective_κzs .< ub)

    # Test that parameters change
    @test convective_κzs[1] != convective_κzs[2]

    iterate!(eki; iterations = 1, show_progress = false)

    @test length(eki.iteration_summaries) == iterations + 2
    @test eki.iteration == iterations + 1

    #####
    ##### Test Resampler
    #####

    resampler = NaNResampler(; abort_fraction=1.0, distribution=FullEnsembleDistribution())

    θ = rand(Nparams, Nensemble)
    p1 = deepcopy(θ[:, 1])
    p2 = deepcopy(θ[:, 2])

    # Fake a forward map output with NaNs
    G = eki.inverting_forward_map(θ)
    G[:, 2] .= NaN

    resample!(resampler, G, θ, eki)

    @test any(isnan.(G)) == false
    @test θ[:, 1] == p1
    @test θ[:, 2] != p2

end