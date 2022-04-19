module PseudoSteppingSchemes

export adaptive_step_parameters

using LineSearches
using Statistics
using LinearAlgebra
using Distributions
using GaussianProcesses

using ..EnsembleKalmanInversions: step_parameters
using ParameterEstimocean.Transformations: ZScore, normalize!, inverse_normalize!

import ..EnsembleKalmanInversions: adaptive_step_parameters, eki_objective

export Constant, Default, GPLineSearch, ConstantConvergence, Kovachki2018, Iglesias2021, Chada2021

# Default pseudo_stepping::Nothing --- it's not adaptive
eki_update(::Nothing, Xₙ, Gₙ, eki, Δtₙ) = eki_update(Constant(Δtₙ), Xₙ, Gₙ, eki)

eki_update(pseudo_scheme, Xₙ, Gₙ, eki, Δtₙ) = eki_update(pseudo_scheme, Xₙ, Gₙ, eki)

function obs_noise_mean(eki)
    μ_noise = zeros(length(eki.mapped_observations))
    μ_noise = eki.tikhonov ? vcat(μ_noise, eki.precomputed_arrays[:μθ]) :
                             μ_noise
    return μ_noise
end

observations(eki) = eki.tikhonov ? eki.precomputed_arrays[:y_augmented] : eki.mapped_observations
obs_noise_covariance(eki) = eki.tikhonov ? eki.precomputed_arrays[:Σ] : eki.noise_covariance
inv_obs_noise_covariance(eki) = eki.tikhonov ? eki.precomputed_arrays[:inv_Σ] : 
                                               eki.precomputed_arrays[:inv_Γy]

function adaptive_step_parameters(pseudo_scheme, Xₙ, Gₙ, eki; Δt=1.0, 
                                    covariance_inflation = 0.0,
                                    momentum_parameter = 0.0)

    N_param, N_ens = size(Xₙ)
    X̅ = mean(Xₙ, dims=2)

    # Forward map augmentation for Tikhonov regularization 
    if eki.tikhonov
        Gₙ = vcat(Gₙ, Xₙ)
    end
    
    Xₙ₊₁, Δtₙ = eki_update(pseudo_scheme, Xₙ, Gₙ, eki, Δt)

    # Apply momentum Xₙ ← Xₙ + λ(Xₙ - Xₙ₋₁)
    @. Xₙ₊₁ = Xₙ₊₁ + momentum_parameter * (Xₙ₊₁ - Xₙ)

    # Apply covariance inflation
    @. Xₙ₊₁ = Xₙ₊₁ + (Xₙ₊₁ - X̅) * covariance_inflation

    return Xₙ₊₁, Δtₙ
end

function step_parameters(X, G, y, Γy, process; Δt=1.0)
    ekp = EnsembleKalmanProcess(X, y, Γy, process; Δt)
    update_ensemble!(ekp, G)
    return get_u_final(ekp), 1.0
end

function iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ=1.0, perturb_observation=false)

    N_obs, N_ens = size(Gₙ)

    y = observations(eki)
    Γy = obs_noise_covariance(eki)
    μ_noise = obs_noise_mean(eki)

    # Scale noise Γy using Δt. 
    Δt⁻¹Γy = Γy / Δtₙ

    y_perturbed = zeros(length(y), N_ens)
    y_perturbed .= y
    if perturb_observation
        ξₙ = rand(MvNormal(μ_noise, Δt⁻¹Γy), N_ens)
        y_perturbed .+= ξₙ # [N_obs x N_ens]
    end

    Cᶿᵍ = cov(Xₙ, Gₙ, dims = 2, corrected = false) # [N_par × N_obs]
    Cᵍᵍ = cov(Gₙ, Gₙ, dims = 2, corrected = false) # [N_obs × N_obs]

    # EKI update: θ ← θ + Cᶿᵍ(Cᵍᵍ + h⁻¹Γy)⁻¹(y + ξₙ - g)
    tmp = (Cᵍᵍ + Δt⁻¹Γy) \ (y_perturbed - Gₙ) # [N_obs × N_ens]
    Xₙ₊₁ = Xₙ + (Cᶿᵍ * tmp) # [N_par × N_ens]

    return Xₙ₊₁
end

frobenius_norm(A) = sqrt(sum(A .^ 2))

function kovachki_2018_update(Xₙ, Gₙ, eki; Δt₀=1.0)

    y = observations(eki)
    Γy = obs_noise_covariance(eki)

    N_ens = size(Xₙ, 2)
    g̅ = mean(Gₙ, dims = 2)
    Γy⁻¹ = inv_obs_noise_covariance(eki)

    # Fill transformation matrix (D(uₙ))ᵢⱼ = ⟨ G(u⁽ʲ⁾) - g̅, Γy⁻¹(G(u⁽ⁱ⁾) - y) ⟩
    D = zeros(N_ens, N_ens)
    for j = 1:N_ens, i = 1:N_ens
        D[i, j] = dot(Gₙ[:, i] - g̅, Γy⁻¹ * (Gₙ[:, j] - y))
    end

    # Calculate time step Δtₙ₋₁ = Δt₀ / (frobenius_norm(D(uₙ)) + ϵ)
    Δtₙ = Δt₀ / (frobenius_norm(D) + 1e-10)

    # Update
    Xₙ₊₁ = Xₙ - (Δtₙ / N_ens) * Xₙ * D

    return Xₙ₊₁, Δtₙ
end

###
### Fixed and adaptive time stepping schemes
###

abstract type AbstractSteppingScheme end

struct Constant{S} <: AbstractSteppingScheme
    step_size :: S
end

Constant(; step_size=1.0) = Constant(step_size)

struct Default{C} <: AbstractSteppingScheme 
    cov_threshold :: C
end

Default(; cov_threshold=0.01) = Default(cov_threshold)

struct GPLineSearch{L, I} <: AbstractSteppingScheme
    learning_rate :: L
    initial_step_size :: I
end

GPLineSearch(; learning_rate=1e-4, initial_step_size=1e-4) = GPLineSearch(learning_rate, initial_step_size)

struct Chada2021{I, B} <: AbstractSteppingScheme
    initial_step_size :: I
    β                 :: B
end

Chada2021(; initial_step_size=1.0, β=0.0) = Chada2021(initial_step_size, β)

struct Iglesias2021 <: AbstractSteppingScheme end

"""
    ConstantConvergence{T} <: AbstractSteppingScheme
    
- `convergence_ratio` (`Number`): The convergence rate for the EKI adaptive time stepping. Default value 0.7.
                                 If a numerical value is given is 0.7 which implies that the parameter spread 
                                 covariance is decreased to 70% of the parameter spread covariance at the previous
                                 EKI iteration.
"""
struct ConstantConvergence{T} <: AbstractSteppingScheme
    convergence_ratio :: T
end

ConstantConvergence(; convergence_ratio=0.7) = ConstantConvergence(convergence_ratio)

struct Kovachki2018{T} <: AbstractSteppingScheme
    initial_step_size :: T
end

Kovachki2018(; initial_step_size=1.0) = Kovachki2018(initial_step_size)

function eki_update(pseudo_scheme::Constant, Xₙ, Gₙ, eki)

    Δtₙ = pseudo_scheme.step_size
    Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)

    @info "Particles stepped with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

function eki_update(pseudo_scheme::Kovachki2018, Xₙ, Gₙ, eki)

    initial_step_size = pseudo_scheme.initial_step_size
    Xₙ₊₁, Δtₙ = kovachki_2018_update(Xₙ, Gₙ, eki; Δt₀=1.0)

    @info "Particles stepped adaptively with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

function eki_update(pseudo_scheme::Chada2021, Xₙ, Gₙ, eki)

    n = eki.iteration
    initial_step_size = pseudo_scheme.initial_step_size
    Δtₙ = ((n+1) ^ pseudo_scheme.β) * initial_step_size
    Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)

    @info "Particles stepped adaptively with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

function eki_update(pseudo_scheme::Default, Xₙ, Gₙ, eki)

    N_param, N_ensemble = size(Xₙ)
    @assert N_ensemble > N_param "The number of parameters exceeds the ensemble size and so the ensemble covariance matrix
                                  will be singular. Please increase the ensemble size to at least $N_param or choose an 
                                  AbstractSteppingScheme that does not rely on inverting the ensemble convariance matrix."

    Δtₙ₋₁ = eki.pseudo_Δt

    accept_stepsize = false
    Δtₙ = copy(Δtₙ₋₁)

    cov_init = cov(Xₙ, dims = 2)
    det_cov_init = det(cov_init)
    @assert det_cov_init != 0 "Ensemble covariance is singular!"

    while !accept_stepsize

        Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)

        cov_new = cov(Xₙ₊₁, dims = 2)

        if det(cov_new) > pseudo_scheme.cov_threshold * det_cov_init
            accept_stepsize = true
        else
            Δtₙ = Δtₙ / 2
        end
    end

    Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)

    @info "Particles stepped adaptively with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

"""
    trained_gp_predict_function(X, y; standardize_X=true)

Return a trained Gaussian Process given inputs X and outputs y.
# Arguments
- `X` (AbstractArray): size `(N_param, N_train)` array of training points.
- `y` (Vector): size `(N_train,)` array of training outputs.
# Keyword Arguments
- `standardize_X` (Bool): whether to standardize the inputs for GP training and prediction.
# Returns
- `predict` (Function): a function that maps size-`(N_param, N_test)` inputs to `(μ, Γgp)`, 
where `μ` is an `(N_test,)` array of corresponding mean predictions and `Γgp` is the 
prediction covariance matrix.
"""
function trained_gp_predict_function(X, y; standardize_X=true)

    N_param = size(X, 1)

    zscore_X = ZScore(mean(X, dims=2), var(X, dims=2))
    standardize_X && normalize!(X, zscore_X)

    zscore_y = ZScore(mean(y), var(y))
    normalize!(y, zscore_y)

    # log- length scale kernel parameter
    ll = [0.0 for _ in 1:N_param]

    # log- noise kernel parameter
    lσ = 0.0

    kern = Matern(5/2, ll, lσ)
    mZero = MeanZero()
    gp = GP(X, y, mZero, kern, -2.0)

    # Use LBFGS to optimize kernel parameters
    optimize!(gp)

    function predict(X) 
        X★ = X[:,:]
        standardize_X && normalize!(X★, zscore_X)
        μ, Γgp = predict_f(gp, X; full_cov=true)
        inverse_normalize!(μ, zscore_y)
        # inverse standardization has element-wise effect on Γgp
        Γgp .*= zscore_y.σ^2

        @assert all(isfinite.(μ))
        @assert all(isfinite.(Γgp))

        if length(μ) == 1
            return μ[1], Γgp[1]
        end
        return μ, Γgp
    end

    return predict
end

ensemble_array(eki, iter) = eki.iteration_summaries[iter].parameters_unconstrained

function eki_update(pseudo_scheme::GPLineSearch, Xₙ, Gₙ, eki)
    
    N_param, N_ensemble = size(Xₙ)
    @assert N_ensemble > N_param "The number of parameters exceeds the ensemble size and so the ensemble covariance matrix
                                  will be singular. Please increase the ensemble size to at least $N_param or choose an 
                                  AbstractSteppingScheme that does not rely on inverting the ensemble convariance matrix."

    # ensemble covariance
    Cᶿᶿ = cov(Xₙ, dims = 2)

    n = eki.iteration

    if n == 0
        Δtₙ = pseudo_scheme.initial_step_size
       return iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ), Δtₙ
    end

    Xₙ = ensemble_array(eki, n)
    Xₙ₋₁ = ensemble_array(eki, n-1)

    # approximate time derivatives of the particles
    # looking backward. size N_param x N_ensemble
    Ẋ_backward = Xₙ - Xₙ₋₁

    # In the continuous-time limit assuming locally linear G,
    # the EKI dynamic for each individual particle θ
    # becomes a preconditioned gradient descent
    # θ̇ = - Cᶿᶿ ∇Φ(θ), where Φ is the EKI objective
    approx_∇Φ = - Cᶿᶿ \ Ẋ_backward

    Xₙ₊₁_test = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ=1.0)

    Ẋ_forward = Xₙ₊₁_test - Xₙ

    ls = BackTracking(c_1 = pseudo_scheme.learning_rate)

    # X is all samples generated thus far
    X = hcat([ensemble_array(eki, iter) for iter in 0:n]...) 
    y = vcat([sum.(eki.iteration_summaries[iter].objective_values) for iter in 0:n]...)

    not_nan_indices = findall(.!isnan.(y))
    X = X[:, not_nan_indices]
    y = y[not_nan_indices]

    predict = trained_gp_predict_function(X, y)

    αs = []
    αinitial = 1.0

    for j = 1:N_ensemble

        xʲ = Xₙ[:, j:j]

        # search direction looking forward
        # Gʲ = G[:, j]
        # s .= - (1/N_ensemble) .* sum( [dot(G[:, k] - g̅, Γy⁻¹ * (Gʲ - y)) .* Xₙ[:, k] for k = 1:N_ensemble] )
        s = Ẋ_forward[:, j]
        
        ϕ(α) = predict(xʲ .+ α .* s)[1]

        # gradient w.r.t. linesearch step size parameter α
        # (equivalent to directional derivative in the search direction, s)
        dϕ_0 = dot(s, approx_∇Φ[:, j])

        ϕ_0 = sum(eki.iteration_summaries[end].objective_values[j])
        
        # For general linesearch, arguments are (ϕ, dϕ, ϕdϕ, α0, ϕ0,dϕ0)
        α, _ = ls(ϕ, αinitial, ϕ_0, dϕ_0)

        push!(αs, α)
    end

    Δtₙ = mean(αs)

    Xₙ₊₁ = Xₙ + Δtₙ * Ẋ_forward
    
    @info "Particles stepped adaptively with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

function volume_ratio(Xₙ₊₁, Xₙ)
    Vⁿ⁺¹ = det(cov(Xₙ₊₁, dims=2))
    Vⁿ   = det(cov(Xₙ,   dims=2))
    return Vⁿ⁺¹ / Vⁿ
end

function eki_update(pseudo_scheme::ConstantConvergence, Xₙ, Gₙ, eki)

    N_param, N_ensemble = size(Xₙ)
    @assert N_ensemble > N_param "The number of parameters exceeds the ensemble size and so the ensemble covariance matrix
                                  will be singular. Please increase the ensemble size to at least $N_param or choose an 
                                  AbstractSteppingScheme that does not rely on inverting the ensemble convariance matrix."

    conv_rate = pseudo_scheme.convergence_ratio

    # Test step forward
    Δtₙ = 1.0
    Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)
    r = volume_ratio(Xₙ₊₁, Xₙ)

    # "Accelerated" fixed point iteration to adjust step_size
    p = 1.1
    iter = 1
    while !isapprox(r, conv_rate, atol=0.03, rtol=0.1) && iter < 10
        Δtₙ *= (r / conv_rate)^p
        Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)
        r = volume_ratio(Xₙ₊₁, Xₙ)
        iter += 1
    end

    @info "Particles stepped adaptively with convergence rate $r (target $conv_rate)"

    return Xₙ₊₁, Δtₙ
end

"""
    eki_update(pseudo_scheme::Iglesias2021, Xₙ, Gₙ, eki)

Implements an EKI update with adaptive time steps based on Iglesias et al. "Adaptive 
Regularization for Ensemble Kalman Inversion," Inverse Problems, 2021.
"""
function eki_update(pseudo_scheme::Iglesias2021, Xₙ, Gₙ, eki)

    n = eki.iteration
    M, J = size(Gₙ)

    Φ = [sum(eki_objective(eki, Xₙ[:, j], Gₙ[:, j], augmented = eki.tikhonov)) for j=1:J]
    Φ_mean = mean(Φ)
    Φ_var = var(Φ)

    qₙ = maximum( (M/(2Φ_mean), sqrt(M/(2Φ_var))) )
    tₙ = n == 0 ? 0.0 : sum(getproperty.(eki.iteration_summaries, :pseudo_Δt))

    Δtₙ = minimum([qₙ, 1-tₙ])
    Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)

    @info "Particles stepped adaptively with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

end # module