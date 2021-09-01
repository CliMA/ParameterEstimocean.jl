# Calibration algorithms
include("line_search_gradient_descent.jl")
include("ensemble_kalman_inversion.jl")
include("ensemble_kalman_inversion_many_columns.jl")
include("simulated_annealing.jl")

function nelder_mead(loss, initial_parameters)
    r = Optim.optimize(loss, [initial_parameters...])
    params = Optim.minimizer(r)
    return params
end

function l_bfgs(loss, initial_parameters)
    r = Optim.optimize(loss, [initial_parameters...], LBFGS())
    params = Optim.minimizer(r)
    return params
end

function random_plugin(loss, initial_parameters; function_calls=1000)
    bounds, _ = get_bounds_and_variance(initial_parameters)
    priors = [Uniform(b...) for b in bounds]
    method = RandomPlugin(priors, function_calls)
    minparam = optimize(loss, method; printresult=false)
    return minparam
end

function gradient_descent(loss, initial_parameters; linebounds = (0, 100.0), linesearches = 100)
    ∇loss(params) = gradient(loss, params) # numerical gradient
    method  = RandomLineSearch(linebounds = linebounds, linesearches = linesearches)
    bestparam = optimize(loss, ∇loss, [initial_parameters...], method);
    return bestparam
end
