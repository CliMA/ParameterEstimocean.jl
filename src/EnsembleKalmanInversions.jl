module EnsembleKalmanInversions

using Distributions
using OffsetArrays
using ProgressBars
using Random
using Printf
using LinearAlgebra
using Suppressor: @suppress
using Statistics
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

using EnsembleKalmanProcesses.EnsembleKalmanProcessModule: sample_distribution

using ..InverseProblems: n_ensemble, observation_map, forward_map, tupify_parameters

#####
##### Priors
#####

function lognormal_with_mean_std(mean, std)
    k = std^2 / mean^2 + 1
    μ = log(mean / sqrt(k))
    σ = sqrt(log(k))
    return LogNormal(μ, σ)
end

struct ConstrainedNormal{FT}
    # θ is the original constrained paramter, θ̃ is the unconstrained parameter ~ N(μ, σ)
    # θ = lower_bound + (upper_bound - lower_bound)/（1 + exp(θ̃)）
    μ :: FT
    σ :: FT
    lower_bound :: FT
    upper_bound :: FT
end

# Scaling factor to give the parameter a magnitude of one
sf(prior) = 1 / abs(prior.μ)

# Model priors are sometimes constrained; EKI deals with unconstrained, Normal priors.
convert_prior(prior::LogNormal) = Normal(sf(prior) * prior.μ, sf(prior) * prior.σ)
convert_prior(prior::Normal) = Normal(sf(prior) * prior.μ, sf(prior) * prior.σ)
convert_prior(prior::ConstrainedNormal) = Normal(prior.μ, prior.σ)

# Convert parameters to unconstrained for EKI
forward_parameter_transform(prior::LogNormal, parameter) = log(parameter^sf(prior))
forward_parameter_transform(prior::Normal, parameter) = parameter * sf(prior)
forward_parameter_transform(cn::ConstrainedNormal, parameter) =
    log((cn.upper_bound - parameter) / (cn.upper_bound - cn.lower_bound))

# Convert parameters from unconstrained (EKI) to constrained
inverse_parameter_transform(prior::LogNormal, parameter) = exp(parameter / sf(prior))
inverse_parameter_transform(prior::Normal, parameter) = parameter / sf(prior)
inverse_parameter_transform(cn::ConstrainedNormal, parameter) =
    cn.lower_bound + (cn.upper_bound - cn.lower_bound) / (1 + exp(parameter))

# Convenience vectorized version
inverse_parameter_transform(priors::NamedTuple, parameters::Vector) =
    NamedTuple(name => inverse_parameter_transform(priors[name], parameters[i])
               for (i, name) in enumerate(keys(priors)))

# Convert covariance from unconstrained (EKI) to constrained
inverse_covariance_transform(::Tuple{Vararg{LogNormal}}, parameters, covariance) =
    Diagonal(exp.(parameters)) * covariance * Diagonal(exp.(parameters))

inverse_covariance_transform(::Tuple{Vararg{Normal}}, parameters, covariance) = covariance

function inverse_covariance_transform(cn::Tuple{Vararg{ConstrainedNormal}}, parameters, covariance)
    upper_bound = [cn[i].upper_bound for i = 1:length(cn)]
    lower_bound = [cn[i].lower_bound for i = 1:length(cn)]
    dT = Diagonal(@. -(upper_bound - lower_bound) * exp(parameters) / (1 + exp(parameters))^2)
    return dT * covariance * dT'
end

mutable struct EnsembleKalmanInversion{I, P, E, M, O, F, S, D}
    inverse_problem :: I
    parameter_distribution :: P
    ensemble_kalman_process :: E
    mapped_observations :: M
    noise_covariance :: O
    inverting_forward_map :: F
    iteration :: Int
    iteration_summaries :: S
    dropped_ensemble_members :: D
end

"""
    parameter_ensemble(eki::EnsembleKalmanInversion)

Return a `Vector` of parameter sets (in physical / constrained space) for each ensemble member.
"""
function parameter_ensemble(eki::EnsembleKalmanInversion)
    priors = eki.inverse_problem.free_parameters.priors
    return parameter_ensemble(eki.ensemble_kalman_process, priors)
end

function parameter_ensemble(ensemble_kalman_process, priors)
    unconstrained_parameters = get_u_final(ensemble_kalman_process) # (N_params, ensemble_size) array
    ensemble_size = size(unconstrained_parameters, 2)
    return [inverse_parameter_transform(priors, unconstrained_parameters[:, n]) for n = 1:ensemble_size]
end

Base.show(io::IO, eki::EnsembleKalmanInversion) =
    print(io, "EnsembleKalmanInversion", '\n',
              "├── inverse_problem: ", typeof(eki.inverse_problem).name.wrapper, '\n',
              "├── parameter_distribution: ", typeof(eki.parameter_distribution).name.wrapper, '\n',
              "├── ensemble_kalman_process: ", typeof(eki.ensemble_kalman_process), '\n',
              "├── mapped_observations: ", summary(eki.mapped_observations), '\n',
              "├── noise_covariance: ", summary(eki.noise_covariance), '\n',
              "├── inverting_forward_map: ", typeof(eki.inverting_forward_map).name.wrapper, '\n',
              "├── iteration: $(eki.iteration)", '\n',
              "└── dropped_ensemble_members: $(eki.dropped_ensemble_members)")

construct_noise_covariance(noise_covariance::AbstractMatrix, y) = noise_covariance

function construct_noise_covariance(noise_covariance::Number, y)
    # Independent noise for synthetic observations
    n_obs = length(y)
    return noise_covariance * Matrix(I, n_obs, n_obs)
end

"""
    EnsembleKalmanInversion(inverse_problem; noise_covariance=1e-2)

 Return an object that interfaces with [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)
and uses Ensemble Kalman Inversion to iteratively "solve" the inverse problem:

```math
y = G(θ) + η,
```

for the parameters ``θ``, where ``y`` is a "normalized" vector of observations,
``G(θ)`` is a forward map that predicts the observations, and ``η ∼ N(0, Γ_y)`` is zero-mean
random noise with covariance matrix ``Γ_y`` representing uncertainty in the observations.

By "solve", we mean that the iteration finds the parameter values ``θ`` that minimizes the
distance between ``y`` and ``G(θ)``.

The "forward map output" `G` can have many interpretations. The specific statistics that `G` computes
have to be selected for each use case to provide a concise summary of the complex model solution that
contains the values that we would most like to match to the corresponding truth values `y`. For example,
in the context of an ocean-surface boundary layer parametrization, this summary could be a vector of 
concatenated `u`, `v`, `b`, `e` profiles at all or some time steps of the CATKE solution.

(For more details on the Ensemble Kalman Inversion algorithm refer to the
[EnsembleKalmanProcesses.jl Documentation](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/ensemble_kalman_inversion/).)

Arguments
=========

- `inverse_problem :: InverseProblem`: Represents an inverse problem representing the comparison between
                                       synthetic observations generated by [Oceananigans.jl](https://clima.github.io/OceananigansDocumentation/stable/)
                                       and model predictions, also generated by Oceananigans.jl.

- `noise_covariance` (`AbstractMatrix` or `Number`): normalized covariance representing observational
                                                     uncertainty. If `noise_covariance isa Number` then
                                                     it's converted to an identity matrix scaled by
                                                     `noise_covariance`.
"""
function EnsembleKalmanInversion(inverse_problem; noise_covariance=1e-2)

    free_parameters = inverse_problem.free_parameters
    original_priors = free_parameters.priors

    transformed_priors = [Parameterized(convert_prior(prior)) for prior in original_priors]
    no_constraints = [[no_constraint()] for _ in transformed_priors]

    parameter_distribution = ParameterDistribution(transformed_priors,
                                                   no_constraints,
                                                   collect(string.(free_parameters.names)))

    ek_process = Inversion()
    initial_ensemble = sample_distribution(parameter_distribution, n_ensemble(inverse_problem))

    # Build EKP-friendly observations "y" and the covariance matrix of observational uncertainty "Γy"
    y = dropdims(observation_map(inverse_problem), dims=2) # length(forward_map_output) column vector
    Γy = construct_noise_covariance(noise_covariance, y)

    # The closure G(θ) maps (N_params, ensemble_size) array to (length(forward_map_output), ensemble_size)
    function inverting_forward_map(θ)
        θ = parameter_ensemble(ensemble_kalman_process, original_priors)

        return forward_map(inverse_problem, θ)
    end

    ensemble_kalman_process = EnsembleKalmanProcess(initial_ensemble, y, Γy, ek_process)

    eki = EnsembleKalmanInversion(inverse_problem,
                                  parameter_distribution,
                                  ensemble_kalman_process,
                                  y,
                                  Γy,
                                  inverting_forward_map,
                                  0,
                                  OffsetArray([], -1),
                                  Set())

    summary = IterationSummary(eki)
    push!(eki.iteration_summaries, summary)

    return eki
end

"""
    UnscentedKalmanInversion(inverse_problem, prior_mean, prior_cov;
                             noise_covariance = 1e-2, α_reg = 1, update_freq = 0)

Return an object that interfaces with [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl)
and uses Unscented Kalman Inversion to iteratively "solve" the inverse problem:

```math
y = G(θ) + η,
```

for the parameters ``θ``, where ``y`` is a "normalized" vector of observations,
``G(θ)`` is a forward map that predicts the observations, and ``η ∼ N(0, Γ_y)`` is zero-mean
random noise with covariance matrix ``Γ_y`` representing uncertainty in the observations.

By "solve", we mean that the iteration finds the parameter values ``θ`` that minimizes the
distance between ``y`` and ``G(θ)``.

(For more details on the Unscented Kalman Inversion algorithm refer to the
[EnsembleKalmanProcesses.jl Documentation](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/unscented_kalman_inversion/).)
    
Arguments
=========

- `inverse_problem :: InverseProblem`: an inverse problem representing the comparison between 
                                       synthetic observations generated by
                                       [Oceananigans.jl](https://clima.github.io/OceananigansDocumentation/stable/)
                                       and model predictions also generated by Oceananigans.jl.

- `prior_mean :: Vector{Float64}`: prior mean

- `prior_cov :: Matrix{Float64}`: prior covariance

- `noise_covariance :: Float64`: observation error covariance

- `α_reg :: Float64`: regularization parameter toward the prior mean (0 < `α_reg` ≤ 1);
                      default `α_reg=1` implies no regularization

- `update_freq :: IT`: set to 0 when the inverse problem is not identifiable (default), namely the
                       inverse problem has multiple solutions, the covariance matrix will represent
                       only the sensitivity of the parameters, instead of posterior covariance information;
                       set to 1 (or anything > 0) when the inverse problem is identifiable, and 
                       the covariance matrix will converge to a good approximation of the 
                       posterior covariance with an uninformative prior
"""
function UnscentedKalmanInversion(inverse_problem, prior_mean, prior_cov;
                                  noise_covariance = 1e-2, α_reg = 1, update_freq = 0)

    free_parameters = inverse_problem.free_parameters
    original_priors = free_parameters.priors

    transformed_priors = [Parameterized(convert_prior(prior)) for prior in original_priors]
    no_constraints = [[no_constraint()] for _ in transformed_priors]
    parameter_distribution = ParameterDistribution(transformed_priors, no_constraints, collect(string.(free_parameters.names)))

    # Build EKP-friendly observations "y" and the covariance matrix of observational uncertainty "Γy"
    y = dropdims(observation_map(inverse_problem), dims=2) # length(forward_map_output) column vector
    Γy = construct_noise_covariance(noise_covariance, y)

    # The closure G(θ) maps (N_params, ensemble_size) array to (length(forward_map_output), ensemble_size)
    function G(θ)
        batch_size = size(θ, 2)
        inverted_parameters = [inverse_parameter_transform.(values(original_priors), θ[:, i]) for i = 1:batch_size]
        return forward_map(inverse_problem, inverted_parameters)
    end

    ensemble_kalman_process = EnsembleKalmanProcess(y, Γy, Unscented(prior_mean, prior_cov, α_reg, update_freq))

    eki = EnsembleKalmanInversion(inverse_problem,
                                  parameter_distribution,
                                  ensemble_kalman_process,
                                  y,
                                  Γy,
                                  G,
                                  0,
                                  OffsetArray([], -1),
                                  Set())

    summary = IterationSummary(eki)
    push!(eki.iteration_summaries, summary)

    return eki
end

"""
    UnscentedKalmanInversionPostprocess(eki)

Returns
=======

- `mean :: Matrix{Float64}`: `N_iterations` × `N_parameters` mean matrix
- `cov :: Vector{Matrix{Float64}}`: `N_iterations` vector of `N_parameters` × `N_parameters` covariance matrix
- `std :: Matrix{Float64}`: `N_iterations` × `N_parameters` standard deviation matrix
- `err :: Vector{Float64}`: `N_iterations` error array
"""
function UnscentedKalmanInversionPostprocess(eki)
    original_priors = eki.inverse_problem.free_parameters.priors
    θ_mean_raw = hcat(eki.ensemble_kalman_process.process.u_mean...)
    θθ_cov_raw = eki.ensemble_kalman_process.process.uu_cov

    θ_mean = similar(θ_mean_raw)
    θθ_cov = similar(θθ_cov_raw)
    θθ_std_arr = similar(θ_mean_raw)

    for i = 1:size(θ_mean, 2) # number of iterations
        θ_mean[:, i] = inverse_parameter_transform.(values(original_priors), θ_mean_raw[:, i])
        θθ_cov[i] = inverse_covariance_transform(values(original_priors), θ_mean_raw[:, i], θθ_cov_raw[i])

        for j = 1:size(θ_mean, 1) # number of parameters
            θθ_std_arr[j, i] = sqrt(θθ_cov[i][j, j])
        end
    end

    return θ_mean, θθ_cov, θθ_std_arr, eki.ensemble_kalman_process.err
end

"""
    struct IterationSummary{P, M, C, V, E}

Container with information about each iteration of the Ensemble Kalman Process.
"""
struct IterationSummary{P, M, C, V, E}
    parameters :: P # constrained
    ensemble_mean :: M # constrained
    ensemble_cov :: C # constrained
    ensemble_var :: V
    mean_square_errors :: E
end

"""
    IterationSummary(eki, parameters, forward_map)

Return the summary for Ensemble Kalman Process `eki` with free `parameters` and `forward_map`.
"""
function IterationSummary(eki, parameters, forward_map)
    N_observations, N_ensemble = size(forward_map)
    original_priors = eki.inverse_problem.free_parameters.priors

    ensemble_mean = mean(parameters, dims=2)
    constrained_ensemble_mean = inverse_parameter_transform.(values(original_priors), ensemble_mean)
    constrained_ensemble_mean = tupify_parameters(eki.inverse_problem, constrained_ensemble_mean)

    ensemble_covariance = cov(parameters, dims=2)
    constrained_ensemble_covariance = inverse_covariance_transform(values(original_priors), parameters, ensemble_covariance)
    constrained_ensemble_variance = tupify_parameters(eki.inverse_problem, diag(constrained_ensemble_covariance))

    constrained_parameters = inverse_parameter_transform.(values(original_priors), parameters)

    constrained_parameters = [tupify_parameters(eki.inverse_problem, constrained_parameters[:, i])
                              for i = 1:size(constrained_parameters, 2)]

    mean_square_errors = [
        mapreduce((x, y) -> (x - y)^2, +, eki.mapped_observations, view(forward_map, :, m)) / N_observations
        for m = 1:N_ensemble
    ]

    return IterationSummary(constrained_parameters,
                            constrained_ensemble_mean,
                            constrained_ensemble_covariance,
                            constrained_ensemble_variance,
                            mean_square_errors)
end

function IterationSummary(eki, parameters)
    original_priors = eki.inverse_problem.free_parameters.priors

    ensemble_mean = mean(parameters, dims=2)
    constrained_ensemble_mean = inverse_parameter_transform.(values(original_priors), ensemble_mean)
    constrained_ensemble_mean = tupify_parameters(eki.inverse_problem, constrained_ensemble_mean)

    ensemble_covariance = cov(parameters, dims=2)
    constrained_ensemble_covariance = inverse_covariance_transform(values(original_priors), parameters, ensemble_covariance)
    constrained_ensemble_variance = tupify_parameters(eki.inverse_problem, diag(constrained_ensemble_covariance))

    constrained_parameters = inverse_parameter_transform.(values(original_priors), parameters)

    constrained_parameters = [tupify_parameters(eki.inverse_problem, constrained_parameters[:, i])
                              for i = 1:size(constrained_parameters, 2)]

    return IterationSummary(constrained_parameters,
                            constrained_ensemble_mean,
                            constrained_ensemble_covariance,
                            constrained_ensemble_variance,
                            fill(NaN, size(constrained_parameters, 2)))
end

"""
    IterationSummary(eki)

Return the summary for Ensemble Kalman Process `eki` before any iteration.
"""
function IterationSummary(eki)
    parameters = get_u_final(eki.ensemble_kalman_process) # (N_params, ensemble_size) array
    return IterationSummary(eki, parameters)
end

function Base.show(io::IO, is::IterationSummary)
    print(io, "IterationSummary(ensemble = ", length(is.mean_square_errors), ")", '\n',
              "                      ", param_str.(keys(is.ensemble_mean))..., '\n',
              "       ensemble_mean: ", param_str.(values(is.ensemble_mean))..., '\n',
              "   ensemble_variance: ", param_str.(values(is.ensemble_var))..., '\n',
              particle_str.(1:length(is.parameters), is.mean_square_errors, is.parameters)...) 
    return nothing
end

quick_summary(iter, is) = println("Iter $iter ", is.ensemble_mean)

function param_str(p::Symbol)
    p_str = string(p)
    length(p_str) > 9 && (p_str = p_str[1:9])
    return @sprintf("% 9s | ", p_str)
end

param_str(p::Number) = @sprintf("%1.3e | ", p)

particle_str(particle, error, parameters) =
    @sprintf("% 7s particle % 3d: ", " ", particle) *
    string(param_str.(values(parameters))...) *
    @sprintf("error = %.3e", error) * "\n"

function drop_ensemble_member!(eki, member)
    parameter_ensemble = eki.iteration_summary[end].parameters
    new_parameter_ensemble = vcat(parameter_ensemble[1:member-1], parameter_ensemble[member+1:end])
    new_ensemble_kalman_process = EnsembleKalmanProcess(new_parameter_ensemble, eki.mapped_observations, eki.noise_covariance, Inversion())

    push!(eki.dropped_ensemble_members, memeber)
    eki.ensemble_kalman_process = new_ensemble_kalman_process

    return nothing
end

"""
    sample(eki, θ, G, n)

Generate `n` new particles sampled from a multivariate Normal distribution parameterized 
by the ensemble mean and covariance computed based on the `N_θ` × `N_ensemble` ensemble 
array `θ`, under the condition that all `n` particles lead to forward map outputs that
are "stable" (don't include `NaNs`). `G` is the inverting forward map computed on
ensemble `θ`.

Return an `N_θ` × `n` array of new particles, along with the inverting forward 
map output corresponding to the new particles.
"""
function sample(eki, θ, G, n)

    n_params, ens_size = size(θ)
    G_length = size(G, 1)

    μ = [mean(θ, dims=2)...]
    Σ = cov(θ, dims=2)
    ens_dist = MvNormal(μ, Σ)

    found_θ = zeros((n_params, 0))
    found_G = zeros((G_length, 0))

    while size(found_θ, 2) < n
        θ_sample = rand(ens_dist, ens_size)
        G_sample = eki.inverting_forward_map(θ_sample)

        nan_values = vec(mapslices(any, isnan.(G_sample); dims=1))
        success_columns = findall(Bool.(1 .- nan_values))

        found_θ = hcat(found_θ, θ_sample[:, success_columns])
        found_G = hcat(found_G, G_sample[:, success_columns])
    end

    return found_θ[:, 1:n], found_G[:, 1:n]
end

"""
    iterate!(eki::EnsembleKalmanInversion; iterations=1)

Iterate the ensemble Kalman inversion problem `eki` forward by `iterations`.
"""
function iterate!(eki::EnsembleKalmanInversion; iterations = 1)

    for _ in ProgressBar(1:iterations)

        θ = get_u_final(eki.ensemble_kalman_process) # (N_params, ensemble_size) array
        G = eki.inverting_forward_map(θ) # (len(G), ensemble_size)

        # Save the parameter values and mean square error between forward map
        # and observations at the current iteration
        summary = IterationSummary(eki, θ, G)
        eki.iteration += 1
        push!(eki.iteration_summaries, summary)

        # ensemble_size vector of bits indicating whether a NaN occured for each particle
        nan_values = vec(mapslices(any, isnan.(G); dims=1))
        nan_columns = findall(nan_values) # indices of columns with `NaN`s
        nan_count = length(nan_columns)
        nan_percent = 100nan_count / size(θ, 2)

        if nan_percent > 90
            error("The forward map for $(nan_percent)% of particles included NaNs. Consider reducing 
                  the model time step, evolving the model for less time, or narrowing the parameter priors.")
        end

        if nan_percent > 0
            @warn "The forward map for $nan_count particles ($(nan_percent)%) included NaNs. Resampling
                    $nan_count particles from a multivariate Normal distribution parameterized by the
                    ensemble mean and covariance."

            found_θ, found_G = sample(eki, θ, G, nan_count)
            θ[:, nan_columns] .= found_θ
            G[:, nan_columns] .= found_G

            new_process = EnsembleKalmanProcess(θ, eki.mapped_observations, eki.noise_covariance, eki.ensemble_kalman_process.process)
            eki.ensemble_kalman_process = new_process
        end

        update_ensemble!(eki.ensemble_kalman_process, G)
    end

    # Return ensemble mean (best guess for optimal parameters)
    best_parameters = eki.iteration_summaries[end].ensemble_mean

    return tupify_parameters(eki.inverse_problem, best_parameters)
end


end # module
