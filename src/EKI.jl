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

#=
function get_bounds_and_variance(default_parameters; stds_within_bounds = 5)

    SomeFreeParameters = typeof(default_parameters).name.wrapper

    # Set bounds on free parameters
    bounds = SomeFreeParameters([(0.0, 10.0) for p in default_parameters]...)

    for (pname, info) in parameter_guide
        set_if_present!(bounds, pname, info.bounds)
    end

    # if stds_within_bounds = n, n standard deviations to either side of the mean fits between the bounds
    variances = SomeFreeParameters((((bound[2] - bound[1])/(2 * stds_within_bounds))^2 for bound in bounds)...)

    variances = Array(variances)

    return bounds, variances
end
=#

using ProgressBars

"""
    eki(ip::InverseProblem, initial_parameters;
                    set_prior_means_to_initial_parameters = true,
                    noise_level = 10^(-2.0),
                    N_iter = 15,
                    stds_within_bounds = 0.6,
                    informed_priors = false,
                    forward_map_output_type = SqrtLossForwardMapOutput
                )

Given y = G(θ) + η where η ∼ N(0, Γy), Ensemble Kalman Inversion solves the inverse problem: to search for
the unknown θ that minimize the distance between the observation y and the prediction G(θ).
The "forward map output" `G` can have many interpretations.
The specific statistics that `G` computes have to be selected for each use 
case to provide a concise summary of the complex model solution that contains the values that 
we would most like to match to the corresponding truth values `y`. In the OSBL context, this summary 
could be, for example, a vector of concatenated `u`, `v`, `b`, `e` profiles at all or some time steps
 of the CATKE solution.

Arguments
- `loss`: function `f(θ::Vector)` that evaluates the loss on the training data
- `initial_parameters`: (`Vector` or `FreeParameter`) if `informed_priors == true`
and `set_prior_means_to_initial_parameters`, this sets the means of the parameter prior distributions.

Keyword Arguments
- `set_prior_means_to_initial_parameters`: (bool)   If `informed` is true, this argument sets whether
to set the parameter prior means to the center of the parameter bounds or to given initial_parameters
- `noise_level`: (Float) Observation noise level γy where Γy = γy*I
- `N_iter`: (Int) Number of EKI iterations
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
function eki(ip::InverseProblem, initial_parameters;
                set_prior_means_to_initial_parameters = true,
                noise_level = 10^(-2.0),
                N_iter = 15,
                stds_within_bounds = 0.6,
                informed_priors = false,
                forward_map_output_type = SqrtLossForwardMapOutput
            )

    N_ens = n_ensemble(ip.simulation.model.grid)

    ip_priors = ip.free_parameters.priors

    bounds, prior_variances = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);

    prior_means = set_prior_means_to_initial_parameters ? [initial_parameters...] : mean.(bounds)

    prior_distns = [Parameterized(Normal([get_μ_σ²(prior_means[i], prior_variances[i], bounds[i])...]...)) for i in 1:length(bounds)]

    constraints = Array([Array([get_constraint(b),]) for b in bounds])

    if informed_priors
        constraints = Array([Array([get_constraint(b),]) for b in bounds])
        prior_distns = [Parameterized(Normal(0.0, stds_within_bounds)) for i in 1:length(initial_parameters)]
    end

    # Seed for pseudo-random number generator for reproducibility
    Random.seed!(41)

    # Define Prior
    prior_names = String.([propertynames(initial_parameters)...])
    prior = ParameterDistribution(prior_distns, constraints, prior_names)

    G = forward_map_output_type(ip, prior)

    # Loss function minimum
    y  = observation(G) # (dim(G), 1) array

    # Independent noise for synthetic observations
    n_obs = length(y)
    Γy = noise_level * Matrix(I, n_obs, n_obs)

    # ℒ = ip(prior_means)
    # pr = norm((σ²s.^(-1/2)) .* μs)^2
    # obs_noise_level = ℒ / pr
    # if objective_scale_info
    #     println("Approx. scale of L in first term (data misfit) of EKI obj:", ℒ)
    #     println("Approx. scale of second term (prior misfit) of EKI obj:", pr)
    #     println("For equal weighting of data misfit and prior misfit in EKI objective, let obs noise level be about:", obs_noise_level)
    # end

    # (dim(G), N_ens)
    initial_ensemble = construct_initial_ensemble(prior, N_ens;
                                                    rng_seed=rng_seed)

    ekiobj = EnsembleKalmanProcess(initial_ensemble, observation, Γy, Inversion())

    @info "Iterative parameter updates in progress"
    for i = ProgressBar(1:N_iter)
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