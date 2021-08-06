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

include("zscore_normalization.jl")

"""
eki_unidimensional(loss, initial_parameters;
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
function eki_unidimensional(loss, initial_parameters;
                                    set_prior_means_to_initial_parameters = true,
                                    noise_level = 10^(-2.0),
                                    N_ens = 10,
                                    N_iter = 15,
                                    stds_within_bounds = 0.6,
                                    informed_priors = false
                                    )

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
    prior_mean = reshape(get_mean(prior),:)
    prior_cov = get_cov(prior)

    # Loss Function Minimum
    y_obs  = [0.0]

    # Independent noise for synthetic observations
    n_obs = length(y_obs)
    Γy = noise_level * Matrix(I, n_obs, n_obs)
    # noise = MvNormal(zeros(n_obs), Γy)

    # We let Forward map be the loss function evaluation
    G(u) = sqrt(loss(transform_unconstrained_to_constrained(prior, u)))
    # println(loss([initial_parameters...]))

    # model_time_series(parameters, loss)

    # ℒ = ce.calibration.loss(prior_means)
    # println("approx. scale of L in first term (data misfit) of EKI obj:", ℒ)
    # pr = norm((σ²s.^(-1/2)) .* μs)^2
    # println("approx. scale of second term (prior misfit) of EKI obj:", pr)
    # obs_noise_level = ℒ / pr
    # println("for equal weighting of data misfit and prior misfit in EKI objective, let obs noise level be about:", obs_noise_level)

    # (dim(G), N_ens)
    initial_ensemble = construct_initial_ensemble(prior, N_ens;
                                                    rng_seed=rng_seed)

    ekiobj = EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion())

    for i in 1:N_iter
        params_i = get_u_final(ekiobj)
        g_ens = hcat([G(params_i[:,i]) for i in 1:N_ens]...) # (N_cases, N_ens, Nz) array
        update_ensemble!(ekiobj, g_ens)
    end

    # All unconstrained
    params = mean(get_u_final(ekiobj), dims=2) # ensemble mean
    losses = [G([mean(get_u(ekiobj, i), dims=2)...])^2 for i in 1:N_iter] # particle loss
    mean_vars = [diag(cov(get_u(ekiobj, i), dims=2)) for i in 1:N_iter] # ensemble variance for each parameter

    params = transform_unconstrained_to_constrained(prior, params)
    params = [params...] # matrix → vector

    # return params, losses, mean_vars
    return params
end

"""
function eki_multidimensional(loss::BatchedLossFunction, ParametersToOptimize, initial_parameters;
                                    set_prior_means_to_initial_parameters = true,
                                    noise_level = 10^(-2.0),
                                    N_ens = 10,
                                    N_iter = 15,
                                    stds_within_bounds = 0.6,
                                    informed_priors = false,
                                    profile_indices = 14:63)

Given y = G(θ) + η where η ∼ N(0, Γy), Ensemble Kalman Inversion solves the inverse problem: to search for
the unknown θ that minimize the distance between the observation y and the prediction G(θ).
In this ``eki_multidimensional" formulation of EKI, we let the forward map output G compute the concatenated final 
profiles for the predicted `u`, `v`, `b`, and `e` at the final timestep. Thus the truth y corresponds to the 

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
function eki_multidimensional(loss::BatchedLossFunction, ParametersToOptimize, initial_parameters;
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
    prior_mean = reshape(get_mean(prior),:)
    prior_cov = get_cov(prior)

    # z-score normalization for profiles
    normalize_function = get_normalization_functions(loss; data_indices=28:126)

    # Loss Function Minimum
    y_obs  = []
    for simulation in loss.batch
        data = simulation.data
        coarse_data = CenterField(simulation.model.grid)

        for field_name in simulation.loss.fields
            if field_name != :e

                data_field = getproperty(data, field_name)[end]
                set!(coarse_data, data_field)

                # zscore_normalize = normalize_function[field_name]
                zscore_normalize = normalize_function[data.name][field_name]
                truth_profile = zscore_normalize(parent(coarse_data.data))[profile_indices]
                push!(y_obs, truth_profile...)
            end
        end
    end

    n_obs = length(y_obs)
    # Independent noise for synthetic observations
    Γy = noise_level * Matrix(I, n_obs, n_obs)
    # noise = MvNormal(zeros(n_obs), Γy)

    # We let Forward map = concatenated profiles at the final timestep
    function G(u)
        all = []
        parameters = ParametersToOptimize(transform_unconstrained_to_constrained(prior, u))
        for simulation in loss.batch
            data = simulation.data
            output = model_time_series(parameters, simulation)

            # println(parameters)
            # println(output.T[end])

            for field_name in simulation.loss.fields
                if field_name != :e
                    model_field = getproperty(output, field_name)[end].data
                    # zscore_normalize = normalize_function[field_name]
                    zscore_normalize = normalize_function[data.name][field_name]
                    model_profile = zscore_normalize(parent(model_field))[profile_indices]
                    push!(all, model_profile...)
                end
            end
        end
        return all
    end

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
