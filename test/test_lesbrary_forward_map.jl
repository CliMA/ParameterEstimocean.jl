using Test
using LinearAlgebra
using Distributions
using DataDeps
using ParameterEstimocean

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity, MixingLength

using ParameterEstimocean.Parameters: unconstrained_prior, transform_to_constrained

@testset "Forward map evaluation with LESbrary observations" begin
    @testset "Convective-adjustment-based ensemble simulations" begin

        data_path = datadep"two_day_suite_4m/free_convection_instantaneous_statistics.jld2"
        times = [1hours, 2hours]
        observations = SyntheticObservations(data_path; field_names=:b, times, transformation=ZScore())
        Nensemble = 10

        simulation = ensemble_column_model_simulation(observations; Nensemble, tracers=:b,
                                                      closure = ConvectiveAdjustmentVerticalDiffusivity())

        Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
        Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
        N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

        simulation.Δt = 10.0

        Qᵘ .= observations.metadata.parameters.momentum_flux
        Qᵇ .= observations.metadata.parameters.buoyancy_flux
        N² .= observations.metadata.parameters.N²_deep

        priors = (convective_κz = ScaledLogitNormal(bounds=(0.1, 1.0)),
                  background_κz = ScaledLogitNormal(bounds=(0.0, 1e-5)))

        free_parameters = FreeParameters(priors)
        calibration = InverseProblem(observations, simulation, free_parameters)

        θ1 = (convective_κz=0.25, background_κz=5e-5)
        θ2 = (convective_κz=1.00, background_κz=1e-4)

        y = observation_map(calibration)
        G = forward_map(calibration, [θ1, θ1, θ2])

        @test G[:, 1] == G[:, 2]
        @test G[:, 2] != G[:, 3]
        @test all(G[:, 3] == G[:, i] for i = 4:Nensemble)
    end

    @testset "CATKE-based ensemble simulations" begin
        data_path = datadep"two_day_suite_4m/weak_wind_strong_cooling_instantaneous_statistics.jld2"
        times = [1hours, 2hours]
        field_names = (:b, :u, :v, :e)
        transformation = (b=ZScore(), u=ZScore(), v=ZScore(), e=RescaledZScore(0.1)) 
        observations = SyntheticObservations(data_path; field_names, times, transformation)

        catke = CATKEVerticalDiffusivity()

        simulation = ensemble_column_model_simulation(observations;
                                                      Nensemble = 10,
                                                      architecture = CPU(),
                                                      tracers = (:b, :e),
                                                      closure = catke)

        Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
        Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
        N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

        simulation.Δt = 10.0

        Qᵘ .= observations.metadata.parameters.momentum_flux
        Qᵇ .= observations.metadata.parameters.buoyancy_flux
        N² .= observations.metadata.parameters.N²_deep

        C⁻c_default = catke.mixing_length.C⁻c

        priors = (; C⁻c = lognormal(mean=C⁻c_default, std=C⁻c_default/100))

        free_parameters = FreeParameters(priors)
        calibration = InverseProblem(observations, simulation, free_parameters)

        Nensemble = simulation.model.grid.Nx
        names = free_parameters.names
        Nθ = length(names)
        unconstrained_priors = NamedTuple(name => unconstrained_prior(priors[name]) for name in names)

        # Initial sample
        X = [rand(unconstrained_priors[i]) for i=1:Nθ, k=1:Nensemble]

        @test size(X) == (Nθ, Nensemble)

        θ = [transform_to_constrained(priors, X[:, k]) for k=1:Nensemble]

        @test length(θ) == Nensemble
        @test all(length(θ[k]) == Nθ for k=1:Nensemble)

        G = forward_map(calibration, θ)
        y = observation_map(calibration)

        @test size(G, 1) == length(y)
        @test size(G, 2) == Nensemble

        θ1 = θ[1] 
        G = forward_map(calibration, [θ1])
        @test all(G[:, 1] == G[:, i] for i = 2:Nensemble)

        θ2 = θ[2] 
        @test values(θ1) != values(θ2)

        G = forward_map(calibration, [θ1, θ2])
        @test G[:, 1] != G[:, 2]
        @test all(G[:, 2] == G[:, i] for i = 3:Nensemble)

        G = forward_map(calibration, [θ1, θ1, θ2])
        @test G[:, 1] == G[:, 2]
        @test all(G[:, 3] == G[:, i] for i = 4:Nensemble)

        G = forward_map(calibration, [θ1, θ1, θ2, θ2])
        @test G[:, 1] == G[:, 2]
        @test G[:, 2] != G[:, 3]
        @test G[:, 3] == G[:, 4]
        @test all(G[:, 4] == G[:, i] for i = 5:Nensemble)
    end
end
