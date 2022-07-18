#####
##### Resampling
#####

resample!(::Nothing, args...) = nothing

struct Resampler{D}
    only_failed_particles :: Bool
    acceptable_failure_fraction :: Float64
    resample_failure_fraction :: Float64
    distribution :: D
    verbose :: Bool
end

function Resampler(; only_failed_particles = true,
                     acceptable_failure_fraction = 0.8,
                     resample_failure_fraction = 0.2,
                     verbose = false,
                     distribution = FullEnsembleDistribution())

    return Resampler(only_failed_particles,
                     acceptable_failure_fraction,
                     resample_failure_fraction,
                     distribution,
                     verbose)
end

abstract type EnsembleDistribution end

function unconstrained_parameters(X)
    μ = [mean(X, dims=2)...]
    Σ = cov(X, dims=2)
    Σᴴ = Matrix(Hermitian(Σ))
    return MvNormal(μ, Σᴴ)
end

struct FullEnsembleDistribution <: EnsembleDistribution end
(::FullEnsembleDistribution)(X, G, successes) = unconstrained_parameters(X)

struct SuccessfulEnsembleDistribution <: EnsembleDistribution end
(::SuccessfulEnsembleDistribution)(X, G, successes) = unconstrained_parameters(X[:, successes])

function failed_particle_str(θ, k, error=nothing)
    first = string(@sprintf(" particle % 3d: ", k), param_str.(values(θ[k]))...)
    error_str = isnothing(error) ? "" : @sprintf(" error = %.6e", error)
    return string(first, error_str, '\n')
end

"""
    resample!(resampler::Resampler, X, G, eki)
    
Resamples the parameters `X` of the `eki` process based on the number of failed particles.
"""
function resample!(resampler::Resampler, X, G, eki)
    # `Nensemble` vector of bits indicating whether an ensemble member has failed.
    particle_failure = eki.mark_failed_particles(G)
    failures = findall(particle_failure) # indices of failed particles
    Nfailures = length(failures)
    failed_fraction = Nfailures / size(X, 2)

    if failed_fraction > 0
        # Print a nice message
        particles = Nfailures == 1 ? "particle" : "particles"

        priors = eki.inverse_problem.free_parameters.priors
        θ = transform_to_constrained(priors, X)
        failed_parameters_message = string("               ",  param_str.(keys(priors))..., '\n',
                                           (failed_particle_str(θ, k) for k in failures)...)

        @warn("""
              The forward map for $Nfailures $particles ($(100failed_fraction)%) failed.
              The failed particles are:
              $failed_parameters_message
              """)
    end

    if failed_fraction > resampler.acceptable_failure_fraction
        error("The forward map for $Nfailures particles ($(100failed_fraction)%) failed. Consider \n" *
              "    1. Increasing `Resampler.acceptable_failure_fraction` for \n" *
              "         EnsembleKalmanInversion.resampler::Resampler \n" * 
              "    2. Reducing the time-step for `InverseProblem.simulation`, \n" *
              "    3. Evolving `InverseProblem.simulation` for less time \n" *
              "    4. Narrowing `FreeParameters` priors.")

    elseif failed_fraction >= resampler.resample_failure_fraction || !(resampler.only_failed_particles)
        # We are resampling!

        if resampler.only_failed_particles
            Nsample = Nfailures
            replace_columns = failures

        else # resample everything
            Nsample = size(G, 2)
            replace_columns = Colon()
        end

        found_X, found_G = find_successful_particles(eki, X, G, Nsample)

        @info "Replacing columns $replace_columns..."
        view(X, :, replace_columns) .= found_X
        view(G, :, replace_columns) .= found_G

        # Sanity...
        if resampler.verbose && resampler.only_failed_particles # print a helpful message about the failure replacements
            Nobs, Nensemble = size(G)
            y = eki.mapped_observations
            errors = [mapreduce((x, y) -> (x - y)^2, +, y, view(G, :, k)) / Nobs for k in failures]

            priors = eki.inverse_problem.free_parameters.priors
            new_θ = transform_to_constrained(priors, X)

            particle_strings = [failed_particle_str(new_θ, k, errors[i]) for (i, k) in enumerate(failures)]
            failed_parameters_message = string("               ",  param_str.(keys(priors))..., '\n',
                                               particle_strings...)

            @info """
            The replacements for failed particles are
            $failed_parameters_message
            """
        end
    end

    return nothing
end

"""
     find_successful_particles(eki, X, G, Nsample)

Generate `Nsample` new particles sampled from a multivariate Normal distribution parameterized 
by the ensemble mean and covariance computed based on the `Nθ` × `Nensemble` ensemble 
array `θ`, under the condition that all `Nsample` particles produce successful forward map
outputs.

`G` (`size(G) =  Noutput × Nensemble`) is the forward map output produced by `θ`.

Returns `Nθ × Nsample` parameter `Array` and `Noutput × Nsample` forward map output `Array`.
"""
function find_successful_particles(eki, X, G, Nsample)
    Nθ, Nensemble = size(X)
    Noutput = size(G, 1)

    Nfound = 0
    found_X = zeros(Nθ, 0)
    found_G = zeros(Noutput, 0)

    mark_failed_particles = eki.mark_failed_particles
    particle_failure = mark_failed_particles(G)
    successful_particles = findall(.!particle_failure)
    existing_sample_distribution = eki.resampler.distribution(X, G, successful_particles)

    while Nfound < Nsample
        @info "Searching for successful particles (found $Nfound of $Nsample)..."

        # Generate `Nensemble` new samples in unconstrained space.
        # Note that eki.inverse_problem.simulation
        # must run `Nensemble` particles no matter what.
        X_sample = rand(existing_sample_distribution, Nensemble)

        G_sample = inverting_forward_map(eki.inverse_problem, X_sample)

        particle_failure = mark_failed_particles(G_sample)
        success_columns = findall(.!particle_failure)
        @info "    ... found $(length(success_columns)) successful particles."

        found_X = cat(found_X, X_sample[:, success_columns], dims=2)
        found_G = cat(found_G, G_sample[:, success_columns], dims=2)
        Nfound = size(found_X, 2)
    end

    # Restrict found particles to requested size
    return found_X[:, 1:Nsample], found_G[:, 1:Nsample]
end

