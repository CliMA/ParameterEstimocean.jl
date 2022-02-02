using Test
using JLD2
using Statistics
using LinearAlgebra
using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.EnsembleKalmanInversions: iterate!, FullEnsembleDistribution, Resampler, resample!, column_has_nan
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

    lower_bound, upper_bound = [0.9, 1.1]
    priors = (convective_κz = ConstrainedNormal(0.0, 1.0, lower_bound, upper_bound),
              background_κz = ConstrainedNormal(0.0, 1.0, 5e-5, 2e-4))
    
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
    @test all(convective_κzs .> lower_bound)
    @test all(convective_κzs .< upper_bound)

    # Test that parameters change
    @test convective_κzs[1] != convective_κzs[2]

    iterate!(eki; iterations = 1, show_progress = false)

    @test length(eki.iteration_summaries) == iterations + 2
    @test eki.iteration == iterations + 1

    #####
    ##### Test Resampler
    #####

    resampler = Resampler(acceptable_failure_fraction = 1.0,
                          distribution = FullEnsembleDistribution())

    θ = rand(Nparams, Nensemble)
    θ1 = deepcopy(θ[:, 1])
    θ2 = deepcopy(θ[:, 2])
    θ3 = deepcopy(θ[:, 3])

    # Fake a forward map output with NaNs
    G = eki.inverting_forward_map(θ)
    view(G, :, 2) .= NaN
    @test any(isnan.(G)) == true

    @test sum(column_has_nan(G)) == 1
    @test column_has_nan(G)[1] == false
    @test column_has_nan(G)[2] == true
    @test column_has_nan(G)[3] == false

    resample!(resampler, θ, G, eki)

    @test sum(column_has_nan(G)) == 0

    @test any(isnan.(G)) == false
    @test θ[:, 1] == θ1
    @test θ[:, 2] != θ2
    @test θ[:, 3] == θ3

    # Resample all particles, not just failed ones

    resampler = Resampler(acceptable_failure_fraction = 1.0,
                          only_failed_particles = false,
                          distribution = FullEnsembleDistribution())

    θ = rand(Nparams, Nensemble)
    θ1 = deepcopy(θ[:, 1])
    θ2 = deepcopy(θ[:, 2])
    θ3 = deepcopy(θ[:, 3])

    # Fake a forward map output with NaNs
    G = eki.inverting_forward_map(θ)
    Gcopy = deepcopy(G)

    resample!(resampler, θ, G, eki)
    @test G != Gcopy
    @test θ[:, 1] != θ1
    @test θ[:, 2] != θ2
    @test θ[:, 3] != θ3

    # Resample particles with SuccessfulEnsembleDistribution.
    # NaN out 2 or 3 columns so that all particles end up identical
    # after resampling.

    resampler = Resampler(acceptable_failure_fraction = 1.0,
                          only_failed_particles = false,
                          distribution = SuccessfulEnsembleDistribution())

    θ = rand(Nparams, Nensemble)
    θ1 = deepcopy(θ[:, 1])
    θ2 = deepcopy(θ[:, 2])
    θ3 = deepcopy(θ[:, 3])

    # Fake a forward map output with NaNs
    G = eki.inverting_forward_map(θ)
    view(G, :, 1) .= NaN
    view(G, :, 2) .= NaN

    @test sum(column_has_nan(G)) == 2
    @test column_has_nan(G)[1] == true
    @test column_has_nan(G)[2] == true
    @test column_has_nan(G)[3] == false

    resample!(resampler, θ, G, eki)

    @test any(isnan.(G)) == false
    @test θ[:, 1] != θ3
    @test θ[:, 2] != θ3
    @test θ[:, 3] != θ3
end
