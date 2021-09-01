function lognormal_μ_σ²(mean, variance)
    k = variance / mean^2 + 1
    μ = log(mean / sqrt(k))
    σ² = log(k)
    return μ, σ²
end

function get_μ_σ²(mean, variance, bounds)
    if bounds[1] == 0.0
        return lognormal_μ_σ²(mean, variance)
    end
    return mean, variance
end

function get_constraint(bounds)
    if bounds[1] == 0.0
        return bounded_below(0.0)
    end
    return no_constraint()
end

"""
eki_unidimensional(loss::DataSet, initial_parameters;
                                    set_prior_means_to_initial_parameters = true,
                                    noise_level = 10^(-2.0),
                                    N_ens = 10,
                                    N_iter = 15,
                                    stds_within_bounds = 0.6,
                                    informed_priors=true

Given y = G(θ) + η where η ∼ N(0, Γy), Ensemble Kalman Inversion solves the inverse problem: to search for
the unknown θ that minimize the distance between the observation y and the prediction G(θ).
In this ``unidimensional" formulation of EKI, we let the forward map output G be the square root of the
evaluation of the loss function on θ directly.

Arguments
- `loss`: function `f(θ::Vector)` that evaluates the loss on the training data
- `initial_parameters`: (`Vector` or `FreeParameter`) if `informed_priors == true`
and `set_prior_means_to_initial_parameters`, this sets the means of the parameter prior distributions.

Keyword Arguments
- `set_prior_means_to_initial_parameters`: (bool)   If `informed` is true, this argument sets whether
to set the parameter prior means to the center of the parameter bounds or to given initial_parameters
- `noise_level`: (Float) Observation noise level γy where Γy = γy*I
- `N_ens`: (Int) Number of ensemble members, J
- `N_iter`: (Int) Number of EKI iterations, Ns
- `stds_within_bounds`: (Float) If `informed` is `true`, sets the number of (prior) standard
deviations `n` spanned by the parameter bounds where σᵢ = (θmaxᵢ - θminᵢ)/n.
- `variance`: (Float) If `informed` is `false`, sets the prior variance γ.
- `informed_priors`: (bool)
    If `false`,
    - for all parameters that are lower-bounded by zero, the unconstrained priors are set to
    log(θ) ∼ N(0, Γ) where Γ = γI, meaning the constrained priors are set to θ ∼ logN(0, Γ).
    - for all parameters that are unbounded, the unconstrained and constrained priors are set to N(0, Γ).
    - γ is set by `variance` argument
    If `true`,
    - parameter priors are set to desired mean and variances according to `initial_parameters`
    and `stds_within_bounds`
"""
function eki_unidimensional(loss::DataSet, initial_parameters;
                                    set_prior_means_to_initial_parameters = true,
                                    noise_level = 10^(-2.0),
                                    N_iter = 15,
                                    stds_within_bounds = 0.6,
                                    informed_priors = false,
                                    objective_scale_info = false
                                    )

    N_ens = ensemble_size(loss.model)

    bounds, prior_variances = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
    prior_means = set_prior_means_to_initial_parameters ? [initial_parameters...] : mean.(bounds)
    prior_distns = [Parameterized(Normal([get_μ_σ²(prior_means[i], prior_variances[i], bounds[i])...]...)) for i in 1:length(bounds)]
    constraints = Array([Array([get_constraint(b),]) for b in bounds])

    if informed_priors
        constraints = Array([Array([get_constraint(b),]) for b in bounds])
        prior_distns = [Parameterized(Normal(0.0, stds_within_bounds)) for i in 1:length(initial_parameters)]
    end

    # Seed for pseudo-random number generator for reproducibility
    rng_seed = 41
    Random.seed!(rng_seed)

    # Define Prior
    prior_names = String.([propertynames(initial_parameters)...])
    prior = ParameterDistribution(prior_distns, constraints, prior_names)

    # Loss Function Minimum
    y_obs  = [0.0]

    # Independent noise for synthetic observations
    n_obs = length(y_obs)
    Γy = noise_level * Matrix(I, n_obs, n_obs)

    # We let Forward map be the square root of the loss function evaluation
    G(u) = @. sqrt(loss(transform_unconstrained_to_constrained(prior, u)))

    ℒ = loss(prior_means)
    pr = norm((σ²s.^(-1/2)) .* μs)^2
    obs_noise_level = ℒ / pr

    if objective_scale_info
        println("Approx. scale of L in first term (data misfit) of EKI obj:", ℒ)
        println("Approx. scale of second term (prior misfit) of EKI obj:", pr)
        println("For equal weighting of data misfit and prior misfit in EKI objective, let obs noise level be about:", obs_noise_level)
    end

    # (dim(G), N_ens)
    initial_ensemble = construct_initial_ensemble(prior, N_ens;
                                                    rng_seed=rng_seed)

    ekiobj = EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion())

    for i = 1:N_iter
        params_i = get_u_final(ekiobj) # (N_params, N_ens) array
        g_ens = G(params_i) # (dim(G), N_ens)
        update_ensemble!(ekiobj, g_ens)
    end

    # All unconstrained
    params = mean(get_u_final(ekiobj), dims=2) # ensemble mean
    losses = [G([mean(get_u(ekiobj, i), dims=2)...])^2 for i in 1:N_iter] # particle loss
    mean_vars = [diag(cov(get_u(ekiobj, i), dims=2)) for i in 1:N_iter] # ensemble variance for each parameter

    params = transform_unconstrained_to_constrained(prior, params)
    params = [params...] # matrix → vector

    return params, losses, mean_vars
end

#
# Hyperparameter search
#

function loss_reduction(calibration, validation, initial_parameters, kwargs)
    params, losses, mean_vars = eki_unidimensional(calibration, initial_parameters; kwargs...)
    valid_loss_start = validation(initial_parameters)
    valid_loss_final = validation(params)

    @info "parameters: $params"
    @info "final loss / initial loss: $(losses[end] / losses[1])"

    train_loss_reduction = losses[end] / losses[1]
    valid_loss_reduction = valid_loss_final / valid_loss_start
    return train_loss_reduction, valid_loss_reduction
end

## Stds within bounds

function plot_stds_within_bounds(calibration, validation, initial_parameters, directory; xrange=-3:0.25:5)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for stds_within_bounds in xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (stds_within_bounds = stds_within_bounds,))
        loss_reductions[stds_within_bounds] = train_loss_reduction
        val_loss_reductions[stds_within_bounds] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus prior std's within bounds (n_σ)", ylabel="Loss reduction (Final / Initial)", xlabel="prior std's spanned by bound width (n_σ)", legend=false, lw=3)
    plot!(loss_reductions, label="training", color=:purple, lw=4)
    plot!(val_loss_reductions, label="validation", color=:blue, lw=4)
    Plots.savefig(p, directory*"stds_within_bounds.pdf")
    println("loss-minimizing stds within bounds: $(argmin(val_loss_reductions))")
end

## Prior Variance
function plot_prior_variance(calibration, validation, initial_parameters, directory; xrange=0.1:0.1:1.0)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for variance = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (stds_within_bounds = variance,))
        loss_reductions[variance] = train_loss_reduction
        val_loss_reductions[variance] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus prior variance", ylabel="Loss reduction (Final / Initial)", xlabel="Prior variance")
    plot!(loss_reductions, label="training", lw=4, color=:purple)
    plot!(val_loss_reductions, label="validation", lw=4, color=:blue)
    Plots.savefig(p, directory*"variance.pdf")
    v = argmin(var_val_loss_reductions)
    println("loss-minimizing variance: $(v)")
    return v
end

## Number of ensemble members

function plot_num_ensemble_members(calibration, validation, initial_parameters, directory; xrange=1:5:30)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for N_ens = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (N_ens = N_ens,))
        loss_reductions[N_ens] = train_loss_reduction
        val_loss_reductions[N_ens] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus N_ens", xlabel="N_ens", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
    plot!(loss_reductions, label="training")
    plot!(val_loss_reductions, label="validation")
    Plots.savefig(p, directory*"N_ens.pdf")
end

## Observation noise level

function plot_observation_noise_level(calibration, validation, initial_parameters, directory; xrange=-2.0:0.2:3.0)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for log_noise_level = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (noise_level = 10.0^log_noise_level,))
        loss_reductions[log_noise_level] = train_loss_reduction
        val_loss_reductions[log_noise_level] = val_loss_reduction
    end
    p = Plots.plot(title="Loss Reduction versus Observation Noise Level", xlabel="log₁₀(Observation noise level)", ylabel="Loss reduction (Final / Initial)", legend=:topleft)
    plot!(loss_reductions, label="training", lw=4, color=:purple)
    plot!(val_loss_reductions, label="validation", lw=4, color=:blue)
    Plots.savefig(p, directory*"obs_noise_level.pdf")
    nl = argmin(val_loss_reductions)
    println("loss-minimizing obs noise level: $(nl)")
    return 10^nl
end

function plot_prior_variance_and_obs_noise_level(calibration, validation, initial_parameters, directory; vrange=0.40:0.025:0.90, nlrange=-2.5:0.1:0.5)
    Γθs = collect(vrange)
    Γys = 10 .^ collect(nlrange)
    losses = zeros((length(Γθs), length(Γys)))
    counter = 1
    countermax = length(Γθs)*length(Γys)
    for i in 1:length(Γθs)
        for j in 1:length(Γys)
            println("progress $(counter)/$(countermax)")
            Γθ = Γθs[i]
            Γy = Γys[j]
            train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (stds_within_bounds=Γθ, noise_level=Γy, N_iter=10, N_ens=10, informed_priors=false))
            losses[i, j] = val_loss_reduction
            counter += 1
        end
    end
    p = Plots.heatmap(Γys, Γθs, losses, xlabel=L"\Gamma_y", ylabel=L"\Gamma_\theta", size=(250,250), yscale=:log10)
    Plots.savefig(p, directory*"GammaHeatmap.pdf")
    v = Γθs[argmin(losses)[1]]
    nl = Γys[argmin(losses)[2]]
    println("loss-minimizing Γθ: $(v)")
    println("loss-minimizing log10(Γy): $(log10(nl))")
    return v, nl
end
