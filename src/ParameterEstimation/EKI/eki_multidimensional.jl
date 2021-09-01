function get_normalization_functions(loss::LossFunction)

    normalize_function = Dict()

    for data in loss.data_batch
        case = data.name
        normalize_function[case] = Dict()
        fields = data.relevant_fields
        targets = data.targets

        for field in fields
            μ = OceanTurbulenceParameterEstimation.profile_mean(data, field; targets=targets, indices=data_indices)
            σ = mean_std(data, field; targets=targets, indices=data_indices)

            normalize(Φ) = (Φ .- μ) ./ σ
            normalize_function[case][field] = normalize
        end
    end
    return normalize_function
end

"""
function eki_multidimensional(loss::LossFunction, ParametersToOptimize, initial_parameters;
                                    set_prior_means_to_initial_parameters = true,
                                    noise_level = 10^(-2.0),
                                    N_ens = 10,
                                    N_iter = 15,
                                    stds_within_bounds = 0.6,
                                    informed_priors = false,
                                    profile_indices = 14:63)

Given y = G(θ) + η where η ∼ N(0, Γy), Ensemble Kalman Inversion solves the inverse problem: to search for
the unknown θ that minimize the distance between the observation y and the prediction G(θ).

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
function eki_multidimensional(loss::LossFunction, ParametersToOptimize, initial_parameters;
                                    set_prior_means_to_initial_parameters = true,
                                    noise_level = 10^(-2.0),
                                    N_ens = 10,
                                    N_iter = 15,
                                    stds_within_bounds = 0.6,
                                    informed_priors = false,
                                    profile_indices = 14:63)

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

    n_obs = length(y_obs)
    # Independent noise for synthetic observations
    Γy = noise_level * Matrix(I, n_obs, n_obs)
    # noise = MvNormal(zeros(n_obs), Γy)

    # println(normalize_function)
    # println("y_obs: $(y_obs)")
    # println("Initial G(u): $(G(transform_constrained_to_unconstrained(prior, [initial_parameters...])))")

    # Ginitial =  G(transform_constrained_to_unconstrained(prior, [initial_parameters...]))
    # x = 1:length(y_obs)
    # a = Plots.plot(x, y_obs[x], label="Observation", color=:red, lw=10, la=0.3)
    # Plots.plot!(x, Ginitial[x], label="G(θ)", color=:red, lw=2, la=1.0, size=(1000,250), legend=:topleft)
    # Plots.savefig("EKI/forward_map_output.pdf")
    #
    # a = Plots.plot(x, y_obs[x], label="Observation", color=:red, lw=2, la=1.0)
    # b = Plots.plot(x, Ginitial[x], label="G(θ)", color=:red, lw=2, la=1.0, legend=:topleft)
    # layout = @layout [c; d]
    # Plots.plot(a,b,size=(1000,500), margins=5*Plots.mm, layout = layout, xlabel="")
    # Plots.savefig("EKI/forward_map_output2.pdf")

    # Calibrate
    initial_ensemble = construct_initial_ensemble(prior, N_ens;
                                                    rng_seed=rng_seed)

    ekiobj = EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion())

    for i in 1:N_iter
        println("Iteration $(i)/$(N_iter) ($(round(i*1000/N_iter)/10)%)")
        params_i = get_u_final(ekiobj)
        g_ens = hcat([G(params_i[:,i]) for i in 1:N_ens]...)
        update_ensemble!(ekiobj, g_ens)
    end

    # All unconstrained
    params = mean(get_u_final(ekiobj), dims=2)
    # losses = [G([mean(get_u(ekiobj, i), dims=2)...])^2 for i in 1:N_iter]
    # mean_vars = [mean(sum((get_u_at_iteration(i) .- params).^2, dims=1)) for i in 1:N_iter]
    # mean_vars = [diag(cov(get_u(ekiobj, i), dims=2)) for i in 1:N_iter]

    # All unconstrained
    params = transform_unconstrained_to_constrained(prior, params)
    params = [params...] # matrix → vector

    return params
end
