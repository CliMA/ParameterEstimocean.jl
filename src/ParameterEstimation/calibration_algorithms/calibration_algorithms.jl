# Calibration algorithms
include("line_search_gradient_descent.jl")
include("ensemble_kalman_inversion.jl")
include("simulated_annealing.jl")

function nelder_mead(nll, initial_parameters)
    r = Optim.optimize(nll, [initial_parameters...])
    params = Optim.minimizer(r)
    return params
end

function l_bfgs(nll, initial_parameters)
    r = Optim.optimize(nll, [initial_parameters...], LBFGS())
    params = Optim.minimizer(r)
    return params
end

function random_plugin(nll, initial_parameters; function_calls=1000)
    bounds, _ = get_bounds_and_variance(initial_parameters)
    priors = [Uniform(b...) for b in bounds]
    method = RandomPlugin(priors, function_calls)
    minparam = optimize(nll, method; printresult=false)
    return minparam
end

function gradient_descent(nll, initial_parameters; linebounds = (0, 100.0), linesearches = 100)
    ∇loss(params) = gradient(nll, params) # numerical gradient
    method  = RandomLineSearch(linebounds = linebounds, linesearches = linesearches)
    bestparam = optimize(nll, ∇loss, [initial_parameters...], method);
    return bestparam
end
