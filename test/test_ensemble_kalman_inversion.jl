using Test
using JLD2
using Statistics
using LinearAlgebra

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans: fields

using ParameterEstimocean
using ParameterEstimocean.EnsembleKalmanInversions: iterate!, pseudo_step!
using ParameterEstimocean.EnsembleKalmanInversions: FullEnsembleDistribution, Resampler
using ParameterEstimocean.EnsembleKalmanInversions: resample!
using ParameterEstimocean.InverseProblems: inverting_forward_map
using ParameterEstimocean.PseudoSteppingSchemes

data_path = "convective_adjustment_test.jld2"
Nensemble = 3
architecture = CPU()

@testset "EnsembleKalmanInversions tests" begin
    @info "  Testing EnsembleKalmanInversion..."

    #####
    ##### Build two InverseProblem
    #####

    observation = SyntheticObservations(data_path, field_names=(:u, :v, :b))

    file = jldopen(observation.path)
    closure = file["serialized/closure"]
    close(file)

    simulation = ensemble_column_model_simulation(observation;
                                                  Nensemble,
                                                  architecture,
                                                  closure,
                                                  tracers = :b)

    batched_observations = BatchedSyntheticObservations([observation, observation], weights=[0.5, 1.0])
    batched_simulation = ensemble_column_model_simulation(batched_observations;
                                                          Nensemble,
                                                          architecture,
                                                          closure,
                                                          tracers = :b)

    for sim in [simulation, batched_simulation]
        sim.model.velocities.u.boundary_conditions.top.condition .= observation.metadata.parameters.Qᵘ
        sim.model.tracers.b.boundary_conditions.top.condition .= observation.metadata.parameters.Qᵇ
        sim.model.tracers.b.boundary_conditions.bottom.condition .= observation.metadata.parameters.N²
        sim.Δt = observation.metadata.parameters.Δt
    end

    lower_bound, upper_bound = bounds = [0.9, 1.1]
    priors = (convective_κz = ScaledLogitNormal(; bounds),
              background_κz = ScaledLogitNormal(bounds=(5e-5, 2e-4)))
    
    Nparams = length(priors)
    
    free_parameters = FreeParameters(priors)
    calibration = InverseProblem(observation, simulation, free_parameters)
    batched_calibration = InverseProblem(batched_observations, batched_simulation, free_parameters)

    #####
    ##### Test EKI
    #####

    for eki in [EnsembleKalmanInversion(calibration; pseudo_stepping=ConstantConvergence(0.9)),
                EnsembleKalmanInversion(batched_calibration; pseudo_stepping=ConstantConvergence(0.9))]

        batch_str = string("(Nbatch = ", size(eki.inverse_problem.simulation.model.grid, 2), ")")
        @testset "EnsembleKalmanInversions construction and iteration tests $batch_str" begin
            @info "  Testing EnsembleKalmanInversion construction and basic iteration $batch_str..."

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

            pseudo_step!(eki)

            @test length(eki.iteration_summaries) == iterations + 2
            @test eki.iteration == iterations + 1
        end

        #####
        ##### Test PseudoSteppingSchemes
        #####

        @testset "PseudoSteppingSchemes tests" begin
            @info "  Testing pseudo-stepping schemes with default hyperparameters"

            for pseudo_stepping in [ConstantPseudoTimeStep, ThresholdedConvergenceRatio, Chada2021, ConstantConvergence, Kovachki2018, Kovachki2018InitialConvergenceRatio, Iglesias2021]
                iterate!(eki; iterations = 1, show_progress=false, pseudo_stepping = pseudo_stepping())
            end
        end

        #####
        ##### Test Resampler
        #####

        @testset "Resampler tests $batch_str" begin
            @info "  Testing resampling and NaN handling $batch_str"

            # Test resample!
            resampler = Resampler(acceptable_failure_fraction = 1.0,
                                  distribution = FullEnsembleDistribution())

            θ = rand(Nparams, Nensemble)
            θ1 = deepcopy(θ[:, 1])
            θ2 = deepcopy(θ[:, 2])
            θ3 = deepcopy(θ[:, 3])

            # Fake a forward map output with NaNs
            norm_exceeds_median = NormExceedsMedian(Inf)
            G = inverting_forward_map(eki.inverse_problem, θ)
            view(G, :, 2) .= NaN
            @test any(isnan.(G)) == true

            @test sum(norm_exceeds_median(G)) == 1
            @test norm_exceeds_median(G)[1] == false
            @test norm_exceeds_median(G)[2] == true
            @test norm_exceeds_median(G)[3] == false

            resample!(resampler, θ, G, eki)

            @test sum(norm_exceeds_median(G)) == 0

            @test any(isnan.(G)) == false
            @test θ[:, 1] == θ1
            @test θ[:, 2] != θ2
            @test θ[:, 3] == θ3

            # Test that model fields get overwritten without NaN
            G = inverting_forward_map(eki.inverse_problem, θ)

            # Particle 2
            view(G, :, 2) .= NaN
            model = eki.inverse_problem.simulation.model
            time_series_collector = eki.inverse_problem.time_series_collector

            # Fill one batch...
            for field_name in keys(fields(model))
                field = fields(model)[field_name]
                collector = time_series_collector.field_time_serieses[field_name]

                field2 = view(parent(field), 2, 1, :) 
                collector2 = view(parent(collector), 2, 1, :, :) 
                fill!(field2, NaN)
                fill!(collector2, NaN)
            end

            @test any(isnan.(model.tracers.b))       
            @test any(isnan.(time_series_collector.field_time_serieses.b))       
            @test any(isnan.(G))

            resample!(resampler, θ, G, eki)

            @test !any(isnan.(model.tracers.b))
            @test !any(isnan.(time_series_collector.field_time_serieses.b))

            # Resample all particles, not just failed ones

            resampler = Resampler(acceptable_failure_fraction = 1.0,
                                  only_failed_particles = false,
                                  distribution = FullEnsembleDistribution())

            θ = rand(Nparams, Nensemble)
            θ1 = deepcopy(θ[:, 1])
            θ2 = deepcopy(θ[:, 2])
            θ3 = deepcopy(θ[:, 3])

            # Fake a forward map output with NaNs
            G = inverting_forward_map(eki.inverse_problem, θ)
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
            G = inverting_forward_map(eki.inverse_problem, θ)
            view(G, :, 1) .= NaN
            view(G, :, 2) .= NaN

            @test sum(norm_exceeds_median(G)) == 2
            @test norm_exceeds_median(G)[1]
            @test norm_exceeds_median(G)[2]
            @test !(norm_exceeds_median(G)[3])

            resample!(resampler, θ, G, eki)

            @test !any(isnan.(G))
            @test θ[:, 1] != θ3
            @test θ[:, 2] != θ3
            @test θ[:, 3] != θ3
        end
    end
end

