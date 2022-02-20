struct IterationSummary{P, M, C, V, E}
    parameters :: P     # constrained
    ensemble_mean :: M  # constrained
    ensemble_cov :: C   # constrained
    ensemble_var :: V
    mean_square_errors :: E
    iteration :: Int
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
    constrained_ensemble_variance = tupify_parameters(eki.inverse_problem, diag(constrained_ensemble_covariance))

    constrained_parameters = transform_to_constrained(priors, X)

    if !isnothing(forward_map_output)
        Nobs, Nens= size(forward_map_output)
        y = eki.mapped_observations
        G = forward_map_output
        mean_square_errors = [mapreduce((x, y) -> (x - y)^2, +, y, view(G, :, k)) / Nobs for k = 1:Nens]
    else
        mean_square_errors = nothing
    end

    return IterationSummary(constrained_parameters,
                            constrained_ensemble_mean,
                            constrained_ensemble_covariance,
                            constrained_ensemble_variance,
                            mean_square_errors,
                            eki.iteration)
end

function Base.show(io::IO, is::IterationSummary)
    max_error, imax = findmax(is.mean_square_errors)
    min_error, imin = findmin(is.mean_square_errors)

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

Base.summary(is::IterationSummary) = string("IterationSummary for ", length(is.parameters),
                                            " particles and ", length(keys(is.ensemble_mean)),
                                            " parameters at iteration ", is.iteration)

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

