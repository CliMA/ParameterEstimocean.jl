module PseudoSteppingSchemes

export adaptive_step_parameters

using LineSearches
using Statistics
using LinearAlgebra
using Printf
using Distributions
using GaussianProcesses

using ..EnsembleKalmanInversions: step_parameters
using ParameterEstimocean.Transformations: ZScore, normalize!, denormalize!

import ..EnsembleKalmanInversions: adaptive_step_parameters, eki_objective

export ConstantPseudoTimeStep
export ThresholdedConvergenceRatio
export ConstantConvergence
export Kovachki2018
export Kovachki2018InitialConvergenceRatio
export Iglesias2021
export Chada2021

# If pseudo_stepping::Nothing, it's not adaptive; Δtₙ₊₁ = Δtₙ.
eki_update(::Nothing, Xₙ, Gₙ, eki, Δtₙ) = eki_update(ConstantPseudoTimeStep(Δtₙ), Xₙ, Gₙ, eki)

# Δtₙ₊₁ selected according to `pseudo_stepping`
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
        Δt⁻¹Γyᴴ = Matrix(Hermitian(Δt⁻¹Γy))
        @assert Δt⁻¹Γyᴴ ≈ Δt⁻¹Γy
        ξₙ = rand(MvNormal(μ_noise, Δt⁻¹Γyᴴ), N_ens)
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

function compute_D(Xₙ, Gₙ, eki)
    y = observations(eki)
    Γy = obs_noise_covariance(eki)
    g̅ = mean(Gₙ, dims = 2)
    Γy⁻¹ = inv_obs_noise_covariance(eki)

    # Transformation matrix (D(uₙ))ᵢⱼ = ⟨ G(u⁽ʲ⁾) - g̅, Γy⁻¹(G(u⁽ⁱ⁾) - y) ⟩
    D = transpose(Gₙ .- g̅) * Γy⁻¹ * (Gₙ .- y)

    return D
end

function kovachki_2018_update(Xₙ, Gₙ, eki; Δt₀=1.0, D=nothing)
    N_ens = size(Xₙ, 2)
    D = isnothing(D) ? compute_D(Xₙ, Gₙ, eki) : D

    # Calculate time step Δtₙ₋₁ = Δt₀ / (frobenius_norm(D(uₙ)) + ϵ)
    Δtₙ = Δt₀ / frobenius_norm(D)

    # Update
    Xₙ₊₁ = Xₙ - (Δtₙ / N_ens) * Xₙ * D

    return Xₙ₊₁, Δtₙ
end

#####
##### Fixed and adaptive time stepping schemes
#####

abstract type AbstractSteppingScheme end

struct ConstantPseudoTimeStep{S} <: AbstractSteppingScheme
    step_size :: S
end

ConstantPseudoTimeStep(; step_size=1.0) = ConstantPseudoTimeStep(step_size)

struct ThresholdedConvergenceRatio{C} <: AbstractSteppingScheme 
    cov_threshold :: C
end

ThresholdedConvergenceRatio(; cov_threshold=0.01) = ThresholdedConvergenceRatio(cov_threshold)

struct Chada2021{I, B} <: AbstractSteppingScheme
    initial_step_size :: I
    β                 :: B
end

Chada2021(; initial_step_size=1.0, β=0.0) = Chada2021(initial_step_size, β)

struct Iglesias2021 <: AbstractSteppingScheme end

struct ConstantConvergence{T} <: AbstractSteppingScheme
    convergence_ratio :: T
end

"""
    ConstantConvergence(; convergence_ratio=0.7)
    
Returns the `ConstantConvergence` psuedo-stepping scheme with target `convergence_ratio`.
With `ConstantConvergence`, the ensemble Kalman inversion (EKI) psuedo step size is adjusted
such that the determinant of the parameter covariance is decreased by 70% after one EKI iteration.
"""
ConstantConvergence(; convergence_ratio=0.7) = ConstantConvergence(convergence_ratio)

struct Kovachki2018{T} <: AbstractSteppingScheme
    initial_step_size :: T
end

Kovachki2018(; initial_step_size=1.0) = Kovachki2018(initial_step_size)

mutable struct Kovachki2018InitialConvergenceRatio{T, I} <: AbstractSteppingScheme
    initial_convergence_ratio :: T
    initial_step_size :: I
end

Kovachki2018InitialConvergenceRatio(; initial_convergence_ratio=0.7) = 
    Kovachki2018InitialConvergenceRatio(initial_convergence_ratio, 0.0)

"""
    eki_update(pseudo_scheme::ConstantPseudoTimeStep, Xₙ, Gₙ, eki)

Implements an EKI update with a fixed time step given by `pseudo_scheme.step_size`.
"""

function eki_update(pseudo_scheme::ConstantPseudoTimeStep, Xₙ, Gₙ, eki)
    Δtₙ = pseudo_scheme.step_size
    Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)
    @info "Particles stepped with time step $Δtₙ"
    return Xₙ₊₁, Δtₙ
end

"""
    eki_update(pseudo_scheme::Kovachki2018, Xₙ, Gₙ, eki)

Implements an EKI update with an adaptive time step estimated as suggested in Kovachki et al.
"Ensemble Kalman Inversion: A Derivative-Free Technique For Machine Learning Tasks" (2018).
"""
function eki_update(pseudo_scheme::Kovachki2018, Xₙ, Gₙ, eki)

    initial_step_size = pseudo_scheme.initial_step_size
    Xₙ₊₁, Δtₙ = kovachki_2018_update(Xₙ, Gₙ, eki; Δt₀=initial_step_size)

    @info "Particles stepped adaptively with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

function eki_update(pseudo_scheme::Kovachki2018InitialConvergenceRatio, Xₙ, Gₙ, eki)

    if pseudo_scheme.initial_step_size == 0

        target = pseudo_scheme.initial_convergence_ratio

        D = compute_D(Xₙ, Gₙ, eki)
        det_cov_init = det(cov(Xₙ, dims = 2))
        conv_ratio(Xₙ₊₁) = det(cov(Xₙ₊₁, dims = 2)) / det_cov_init
        
        # First guess
        Δt₀ = 1.0
        Xₙ₊₁, Δtₙ = kovachki_2018_update(Xₙ, Gₙ, eki; Δt₀, D)

        # Coarse adjustment to find the right order of magnitude
        r = conv_ratio(Xₙ₊₁)
        too_big = r > target
        i = too_big
        first_guess(i, Δt₀) = i ? Δt₀*2 : Δt₀/2
        second_guess(i, Δt₀) = first_guess(!i, Δt₀)

        iter = 1
        while i == too_big && iter < 10
            # Keep adjusting Δt₀ until the truth value `i` flips

            # The first guess assumes that the convergence ratio decreases with increasing time step
            Δt₀_guess = first_guess(i, Δt₀)
            Xₙ₊₁, Δtₙ = kovachki_2018_update(Xₙ, Gₙ, eki; Δt₀=Δt₀_guess, D)
            r_test = conv_ratio(Xₙ₊₁)

            if (r_test > r) == i
                # Convergence ratio didn't adjust in the direction we expected; try the other direction
                Δt₀_guess = second_guess(i, Δt₀)
                Xₙ₊₁, Δtₙ = kovachki_2018_update(Xₙ, Gₙ, eki; Δt₀=Δt₀_guess, D)
                r_test = conv_ratio(Xₙ₊₁)
            end
    
            Δt₀ = Δt₀_guess
            r = r_test
            i = r > target
            iter += 1
        end
        
        # Fine-grained adjustment
        p = 1.1
        iter = 1
        while !isapprox(r, target, atol=0.03, rtol=0.1) && iter < 10

            Δt₀_test = Δt₀ * (r / target)^p
            Xₙ₊₁, Δtₙ = kovachki_2018_update(Xₙ, Gₙ, eki; Δt₀=Δt₀_test, D)
            r_test = conv_ratio(Xₙ₊₁)

            # Make sure the convergence ratio moved closer to the target; otherwise halt
            # to prevent divergence.
            if abs(r_test - target) > abs(r - target)
                break
            else
                Δt₀ = Δt₀_test
                r = r_test    
            end
            iter += 1
        end

        pseudo_scheme.initial_step_size = Δt₀

        @info "Particles stepped adaptively with time step $Δtₙ and convergence ratio $r (target $target)."

        return Xₙ₊₁, Δtₙ
    
    else
        return eki_update(Kovachki2018(initial_step_size = pseudo_scheme.initial_step_size), Xₙ, Gₙ, eki)
    end
end

"""
    eki_update(pseudo_scheme::ThresholdedConvergenceRatio, Xₙ, Gₙ, eki; initial_guess=nothing)

Implements an EKI update with an adaptive time step estimated as suggested in Chada, Neil and Tong, Xin 
"Convergence Acceleration of Ensemble Kalman Inversion in Nonlinear Settings," Math. Comp. 91 (2022).
"""
function eki_update(pseudo_scheme::Chada2021, Xₙ, Gₙ, eki)

    n = eki.iteration
    initial_step_size = pseudo_scheme.initial_step_size
    Δtₙ = ((n+1) ^ pseudo_scheme.β) * initial_step_size
    Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)

    @info "Particles stepped adaptively with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

"""
    eki_update(pseudo_scheme::ThresholdedConvergenceRatio, Xₙ, Gₙ, eki; initial_guess=nothing)

Implements an EKI update with an adaptive time step estimated by finding the first step size
in the sequence Δtₖ = Δtₙ₋₁(1/2)^k with k = {0,1,2,...} that satisfies 
|cov(Xₙ₊₁)|/|cov(Xₙ)| > pseudo_scheme.cov_threshold, assuming the determinant ratio
is a monotonically increasing function of k. If an `initial_guess` is provided,
`Δtₙ₋₁` in the above sequence is replaced with `initial_guess`. If an `initial_guess`
is not provided, the time step can only decrease or stay the same at future iterations
with this time stepping scheme.
"""
function eki_update(pseudo_scheme::ThresholdedConvergenceRatio, Xₙ, Gₙ, eki; initial_guess=nothing, report=true)

    N_param, N_ensemble = size(Xₙ)
    @assert N_ensemble > N_param "The number of parameters exceeds the ensemble size and so the ensemble covariance matrix
                                  will be singular. Please increase the ensemble size to at least $N_param or choose an 
                                  AbstractSteppingScheme that does not rely on inverting the ensemble convariance matrix."

    Δtₙ₋₁ = isnothing(initial_guess) ? eki.pseudo_Δt : initial_guess

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

    report && @info "Particles stepped adaptively with time step $Δtₙ"

    return Xₙ₊₁, Δtₙ
end

"""
    trained_gp_predict_function(X, y; standardize_X=true, zscore_limit=nothing, kernel=nothing)

Return a trained Gaussian Process given inputs X and outputs y.
# Arguments
- `X` (AbstractArray): size `(N_param, N_train)` array of training points.
- `y` (Vector): size `(N_train,)` array of training outputs.
# Keyword Arguments
- `standardize_X` (Bool): whether to standardize the inputs for GP training and prediction.
- `zscore_limit` (Int): specifies the number of standard deviations outside of which 
all output entries and their corresponding inputs should be removed from the training data
in an initial filtering step.
- `kernel` (GaussianProcesses.Kernel): kernel to be optimized and used in the GP.
# Returns
- `predict` (Function): a function that maps size-`(N_param, N_test)` inputs to `(μ, Γgp)`, 
where `μ` is an `(N_test,)` array of corresponding mean predictions and `Γgp` is the 
prediction covariance matrix.
"""
function trained_gp_predict_function(X, y; standardize_X=true, zscore_limit=nothing, kernel=nothing)

    X = copy(X)
    y = copy(y)

    if !isnothing(zscore_limit)

        y_temp = copy(y)
        normalize!(y_temp, ZScore(mean(y_temp), std(y_temp)))
        to_keep = findall(x -> (x > -zscore_limit && x < zscore_limit), y_temp)
        y = y[to_keep]
        X = X[:, to_keep]

        n_pruned = length(y_temp) - length(to_keep)

        if n_pruned > 0
            percent_pruned = round((100n_pruned / length(y)); sigdigits=3)


            @info "Pruned $n_pruned GP training points ($percent_pruned%) corresponding to outputs 
                outside $zscore_limit standard deviations from the mean."
        end
    end

    zscore_y = ZScore(mean(y), std(y))
    normalize!(y, zscore_y)

    zscore_X = ZScore(mean(X, dims=2), std(X, dims=2))
    standardize_X && normalize!(X, zscore_X)

    if isnothing(kernel)

        N_param = size(X, 1)

        # log- length scale kernel parameter
        ll = zeros(N_param)
    
        # log- noise kernel parameter
        lσ = 0.0
    
        kernel = SE(ll, lσ)
    end

    mZero = MeanZero()
    gp = GP(X, y, mZero, kernel, -2.0)

    # Use LBFGS to optimize kernel parameters
    optimize!(gp)

    function predict(X) 
        X★ = X[:,:]
        standardize_X && normalize!(X★, zscore_X)
        μ, Γgp = predict_f(gp, X★; full_cov=true)

        denormalize!(μ, zscore_y)
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

ensemble_array(eki, iter) = eki.iteration_summaries[iter].unconstrained_parameters

"""
    eki_update(pseudo_scheme::ConstantConvergence, Xₙ, Gₙ, eki)

Implements an EKI update with an adaptive time step estimated to encourage a prescribed
rate of ensemble collapse as measured by the ratio of the ensemble 
covariance matrix determinants at consecutive iterations.
"""
function eki_update(pseudo_scheme::ConstantConvergence, Xₙ, Gₙ, eki)

    N_param, N_ensemble = size(Xₙ)
    @assert N_ensemble > N_param "The number of parameters exceeds the ensemble size and so the ensemble covariance matrix
                                  will be singular. Please increase the ensemble size to at least $N_param or choose an 
                                  AbstractSteppingScheme that does not rely on inverting the ensemble convariance matrix."

    conv_rate = pseudo_scheme.convergence_ratio

    # Start with Δtₙ = 1.0; `Δtₙ_first_guess` is the first time step in the sequence Δtₖ = (1/2)^k where k={0,1,2...}
    # such that |cov(Xₙ₊₁)|/|cov(Xₙ)| > pseudo_scheme.convergence_ratio (assuming the determinant ratio
    # is monotonically increasing as a function of k).
    _, Δtₙ_first_guess = eki_update(ThresholdedConvergenceRatio(cov_threshold=pseudo_scheme.convergence_ratio),
                                    Xₙ, Gₙ, eki; initial_guess=1.0, report=false)

    # `Δtₙ_first_guess` provides a reasonable initial guess for the time step. If we were to 
    # start the fixed point iteration algorithm below with an initial guess of 1.0, the initial volume 
    # volume ratio could be obscenely small, leading to an obscenely small initial Δtₙ, 
    # sending the subsequent `r` values to ≈1.0. In such a situation the subsequently calculated Δtₙ 
    # would remain tiny, never recovering the desired order of magnitude; `r` would remain ≈1.0.
    # `Δtₙ_first_guess` starts us off in the right order of magnitude for the linear assumption 
    # on `r` vs `Δtₙ` to be fruitful.
    Δtₙ = Δtₙ_first_guess

    det_cov_init = det(cov(Xₙ, dims = 2))

    # Test step forward
    Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)
    r = det(cov(Xₙ₊₁, dims=2)) / det_cov_init

    # "Accelerated" fixed point iteration to adjust step_size
    p = 1.1
    iter = 1
    while !isapprox(r, conv_rate, atol=0.03, rtol=0.1) && iter < 10
        Δtₙ *= (r / conv_rate)^p
        Xₙ₊₁ = iglesias_2013_update(Xₙ, Gₙ, eki; Δtₙ)
        r = det(cov(Xₙ₊₁, dims=2)) / det_cov_init
        iter += 1
    end

    # A nice message
    intro_str       = "Pseudo time step found for ConstantConvergence pseudostepping."
    convergence_str = @sprintf("      ├─ convergence ratio: %.6f (target: %.2f)", r, conv_rate)
    time_step_str   = @sprintf("      └─ psuedo time step: %.3e", Δtₙ)
    @info string(intro_str, '\n', convergence_str, '\n', time_step_str)

    return Xₙ₊₁, Δtₙ
end

"""
    eki_update(pseudo_scheme::Iglesias2021, Xₙ, Gₙ, eki)

Implements an EKI update with an adaptive time step based on Iglesias et al. "Adaptive 
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

    @info "Pseudo time step $Δtₙ found for the Iglesias2021 pseudostepping."

    return Xₙ₊₁, Δtₙ
end

end # module
