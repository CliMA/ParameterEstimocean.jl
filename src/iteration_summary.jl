using ..Parameters: transform_to_unconstrained, build_parameters_named_tuple

using Oceananigans.Utils: prettysummary

struct IterationSummary{P, M, C, V, E, O}
    parameters :: P     # constrained
    ensemble_mean :: M  # constrained
    ensemble_cov :: C   # constrained
    ensemble_var :: V
    mean_square_errors :: E
    objective_values :: O
    iteration :: Int
    pseudotime :: Float64
    pseudo_Δt :: Float64
end

"""
    eki_objective(eki, θ::AbstractVector, G::AbstractVector; constrained = false)

Given forward map `G` and parameters `θ`, return a tuple `(Φ₁, Φ₂)` 
of terms in the ensemble Kalman inversion (`eki`) regularized objective
function, where,

```code
Φ = Φ₁ + Φ₂
```

`Φ₁` measures output misfit `½ || Γy^(-¹/₂) * (y .- G(θ)) ||²` and 
`Φ₂` measures prior misfit `½ || Γθ^(-¹/₂) * (θ .- μθ) ||²`, where `y` is the observation 
map, `G(θ)` is the forward map, `Γy` is the observation noise covariance, `Γθ` is 
the prior covariance, and `μθ` represents the prior means. Note that `Γ^(-¹/₂) = 
inv(sqrt(Γ))`.

When keyword argument `constrained` is provided with `true` then input `θ`
is assumed to represent constrained parameters.
"""
function eki_objective(eki, θ::AbstractVector, G::AbstractVector; constrained = false)
    y = eki.mapped_observations
    Γy = eki.noise_covariance

    fp = eki.inverse_problem.free_parameters
    priors = fp.priors
    unconstrained_priors = [unconstrained_prior(priors[name]) for name in fp.names]
    μθ = getproperty.(unconstrained_priors, :μ)
    Γθ = diagm( getproperty.(unconstrained_priors, :σ).^2 )

    if constrained
        θ = [transform_to_unconstrained(priors[name], θ[i])
                for (i, name) in enumerate(keys(priors))]
    end
    
    # Φ₁ = (1/2)*|| Γy^(-½) * (y - G) ||²
    Φ₁ = (1/2) * norm(inv(sqrt(Γy)) * (y .- G))^2
    # Φ₂ = (1/2)*|| Γθ^(-½) * (θ - μθ) ||² 
    Φ₂ = (1/2) * norm(inv(sqrt(Γθ)) * (θ .- μθ))^2
    return (Φ₁, Φ₂)
end

"""
    IterationSummary(eki, X, forward_map_output=nothing)

Return the summary for ensemble Kalman inversion `eki` 
with unconstrained parameters `X` and `forward_map_output`.
"""
function IterationSummary(eki, X, forward_map_output=nothing)
    priors = eki.inverse_problem.free_parameters.priors

    ensemble_mean = mean(X, dims=2)[:] 
    constrained_ensemble_mean = transform_to_constrained(priors, ensemble_mean)

    ensemble_covariance = cov(X, dims=2)
    constrained_ensemble_covariance = inverse_covariance_transform(values(priors), X, ensemble_covariance)

    constrained_ensemble_variance = build_parameters_named_tuple(eki.inverse_problem.free_parameters,
                                                                 diag(constrained_ensemble_covariance))

    constrained_parameters = transform_to_constrained(priors, X)

    G = forward_map_output
    if !isnothing(forward_map_output)
        Nobs, Nens= size(forward_map_output)
        y = eki.mapped_observations
        mean_square_errors = [mapreduce((x, y) -> (x - y)^2, +, y, view(G, :, k)) / Nobs for k = 1:Nens]
    else
        mean_square_errors = nothing
    end

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    objective_values = [eki_objective(eki, X[:, j], G[:, j]) for j in 1:size(G, 2)]

    return IterationSummary(constrained_parameters,
                            constrained_ensemble_mean,
                            constrained_ensemble_covariance,
                            constrained_ensemble_variance,
                            mean_square_errors,
                            objective_values,
                            eki.iteration,
                            eki.pseudotime,
                            eki.pseudo_Δt)
end

function finitefind(a, val, find)
    finite_a = deepcopy(a)
    finite_a[.!isfinite.(a)] .= val
    return find(finite_a)
end

finitefindmin(a) = finitefind(a, Inf, findmin)
finitefindmax(a) = finitefind(a, -Inf, findmax)

function Base.show(io::IO, is::IterationSummary)
    max_error, imax = finitefindmax(is.mean_square_errors)
    min_error, imin = finitefindmin(is.mean_square_errors)

    names = keys(is.ensemble_mean)
    parameter_matrix = [is.parameters[k][name] for name in names, k = 1:length(is.parameters)]
    min_parameters = minimum(parameter_matrix, dims=2)
    max_parameters = maximum(parameter_matrix, dims=2)

    print(io, summary(is), '\n')

    print(io, "                      ", param_str.(keys(is.ensemble_mean))..., '\n',
              "       ensemble_mean: ", param_str.(values(is.ensemble_mean))..., '\n',
              particle_str("best", is.mean_square_errors[imin], is.parameters[imin]), '\n',
              particle_str("worst", is.mean_square_errors[imax], is.parameters[imax]), '\n',
              "             minimum: ", param_str.(min_parameters)..., '\n',
              "             maximum: ", param_str.(max_parameters)..., '\n',
              "   ensemble_variance: ", param_str.(values(is.ensemble_var))...)

    return nothing
end

Base.summary(is::IterationSummary) = string("IterationSummary(",
                                            "iteration=", is.iteration,
                                            ", pseudotime=", prettysummary(is.pseudotime),
                                            ", pseudo_Δt=", prettysummary(is.pseudo_Δt), ") ",
                                            "for ", length(is.parameters),
                                            " particles and ", length(keys(is.ensemble_mean)),
                                            " parameters")

function param_str(p::Symbol)
    p_str = string(p)
    length(p_str) > 9 && (p_str = p_str[1:9])
    return @sprintf("% 10s | ", p_str)
end

param_str(p::Number) = @sprintf("% -1.3e | ", p)

particle_str(particle, error, parameters) =
    @sprintf("% 11s particle: ", particle) *
    string(param_str.(values(parameters))...) *
    @sprintf("error = %.6e", error)
