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

function ensemble_normal_distribution(X)
    μ = [mean(X, dims=2)...]
    Σ = cov(X, dims=2)
    Σᴴ = Matrix(Hermitian(Σ))
    @assert Σᴴ ≈ Σ 
    return MvNormal(μ, Σᴴ)
end

struct FullEnsembleDistribution <: EnsembleDistribution end
(::FullEnsembleDistribution)(X, G) = ensemble_normal_distribution(X)

struct SuccessfulEnsembleDistribution <: EnsembleDistribution end
(::SuccessfulEnsembleDistribution)(X, G) = ensemble_normal_distribution(X[:, findall(.!column_has_nan(G))])

""" Return a BitVector indicating which particles are NaN."""
column_has_nan(G) = vec(mapslices(any, isnan.(G); dims=1))

function failed_particle_str(θ, k, error=nothing)
    first = string(@sprintf(" particle % 3d: ", k), param_str.(values(θ[k]))...)
    error_str = isnothing(error) ? "" : @sprintf(" error = %.6e", error)
    return string(first, error_str, '\n')
end

"""
    resample!(resampler::Resampler, θ, G, eki)
    
Resamples the parameters `θ` of the `eki` process based on the number of `NaN` values
inside the forward map output `G`.
"""
function resample!(resampler::Resampler, X, G, eki)
    # `Nensemble` vector of bits indicating, for each ensemble member, whether the forward map contained `NaN`s
    nan_values = column_has_nan(G)
    nan_columns = findall(nan_values) # indices of columns (particles) with `NaN`s
    nan_count = length(nan_columns)
    nan_fraction = nan_count / size(X, 2)

    if nan_fraction > 0
        # Print a nice message
        particles = nan_count == 1 ? "particle" : "particles"

        priors = eki.inverse_problem.free_parameters.priors
        θ = transform_to_constrained(priors, X)
        failed_parameters_message = string("               ",  param_str.(keys(priors))..., '\n',
                                           (failed_particle_str(θ, k) for k in nan_columns)...)

        @warn("""
              The forward map for $nan_count $particles ($(100nan_fraction)%) included NaNs.
              The failed particles are:
              $failed_parameters_message
              """)
    end

    if nan_fraction > resampler.acceptable_failure_fraction
        error("The forward map for $nan_count particles ($(100nan_fraction)%) included NaNs. Consider \n" *
              "    1. Increasing `Resampler.acceptable_failure_fraction` for \n" *
              "         EnsembleKalmanInversion.resampler::Resampler \n" * 
              "    2. Reducing the time-step for `InverseProblem.simulation`, \n" *
              "    3. Evolving `InverseProblem.simulation` for less time \n" *
              "    4. Narrowing `FreeParameters` priors.")

    elseif nan_fraction >= resampler.resample_failure_fraction || !(resampler.only_failed_particles)
        # We are resampling!

        if resampler.only_failed_particles
            Nsample = nan_count
            replace_columns = nan_columns

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
            errors = [mapreduce((x, y) -> (x - y)^2, +, y, view(G, :, k)) / Nobs for k in nan_columns]

            priors = eki.inverse_problem.free_parameters.priors
            new_θ = transform_to_constrained(priors, X)

            particle_strings = [failed_particle_str(new_θ, k, errors[i]) for (i, k) in enumerate(nan_columns)]
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
outputs (don't include `NaNs`).

`G` (`size(G) =  Noutput × Nensemble`) is the forward map output produced by `θ`.

Returns `Nθ × Nsample` parameter `Array` and `Noutput × Nsample` forward map output `Array`.
"""
function find_successful_particles(eki, X, G, Nsample)
    Nθ, Nensemble = size(X)
    Noutput = size(G, 1)

    Nfound = 0
    found_X = zeros(Nθ, 0)
    found_G = zeros(Noutput, 0)
    existing_sample_distribution = eki.resampler.distribution(X, G)

    while Nfound < Nsample
        @info "Searching for successful particles (found $Nfound of $Nsample)..."

        # Generate `Nensemble` new samples in unconstrained space.
        # Note that eki.inverse_problem.simulation
        # must run `Nensemble` particles no matter what.
        X_sample = rand(existing_sample_distribution, Nensemble)

        G_sample = inverting_forward_map(eki.inverse_problem, X_sample)

        nan_values = column_has_nan(G_sample)
        success_columns = findall(.!column_has_nan(G_sample))
        @info "    ... found $(length(success_columns)) successful particles."

        found_X = cat(found_X, X_sample[:, success_columns], dims=2)
        found_G = cat(found_G, G_sample[:, success_columns], dims=2)
        Nfound = size(found_X, 2)
    end

    # Restrict found particles to requested size
    return found_X[:, 1:Nsample], found_G[:, 1:Nsample]
end

