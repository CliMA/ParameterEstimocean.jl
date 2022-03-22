resampler = Resampler(resample_failure_fraction=0.5, acceptable_failure_fraction=1.0)

function adaptive_step_parameters(convergence_rate, Xⁿ, Gⁿ, y, Γy, process)
    # Test step forward
    step_size = 1
    Xⁿ⁺¹ = step_parameters(Xⁿ, Gⁿ, y, Γy, process; step_size)
    r = volume_ratio(Xⁿ⁺¹, Xⁿ)

    # "Accelerated" fixed point iteration to adjust step_size
    p = 1.1
    iter = 1
    while !isapprox(r, convergence_rate, atol=0.03, rtol=0.1) && iter < 10
        step_size *= (r / convergence_rate)^p
        Xⁿ⁺¹ = step_parameters(Xⁿ, Gⁿ, y, Γy, process; step_size)
        r = volume_ratio(Xⁿ⁺¹, Xⁿ)
        iter += 1
    end

    @info "Particles stepped adaptively with convergence rate $r (target $convergence_rate)"

    return Xⁿ⁺¹
end

frobenius_norm(A) = sqrt(sum(A .^ 2))

function step_parameters(X, G, y, Γy, 
                        step_size = 1.0, 
                        covariance_inflation = 1.0)

    # X is [N_par × N_ens]
    N_obs = size(G, 1) # N_obs

    X̅ = mean(X, dims=2) # [1 × N_ens]

    # Apply covariance inflation
    @. X = X + (X - X̅) * covariance_inflation

    # Scale noise Γy using Δt. 
    Δt⁻¹Γy = Γy / step_size
    ξₙ = rand(ekp.rng, MvNormal(zeros(N_obs), Γy_scaled), ekp.N_ens)

    y = ekp.obs_mean

    cov_θg = cov(X, G, dims = 2, corrected = false) # [N_par × N_obs]
    cov_gg = cov(G, G, dims = 2, corrected = false) # [N_obs × N_obs]

    # EKI update: θ ← θ + cov_θg(cov_gg + h⁻¹Γy)⁻¹(y + ξₙ - g)
    tmp = (cov_gg + Δt⁻¹Γy) \ (y + ξₙ - G) # [N_obs × N_ens]
    X = X + (cov_θg * tmp) # [N_par × N_ens]  
end



"""
    find_ekp_stepsize(ekp::EnsembleKalmanProcess{FT, IT, Inversion}, g::AbstractMatrix{FT}; cov_threshold::FT=0.01) where {FT}
Find largest stepsize for the EK solver that leads to a reduction of the determinant of the sample
covariance matrix no greater than cov_threshold. 
"""
function find_ekp_stepsize(
    ekp::EnsembleKalmanProcess{FT, IT, Inversion},
    g::AbstractMatrix{FT};
    cov_threshold::FT = 0.01,
) where {FT, IT}
    accept_stepsize = false
    if !isempty(ekp.Δt)
        Δt = deepcopy(ekp.Δt[end])
    else
        Δt = FT(1)
    end
    # final_params [N_par × N_ens]
    cov_init = cov(get_u_final(ekp), dims = 2)
    while accept_stepsize == false
        ekp_copy = deepcopy(ekp)
        update_ensemble!(ekp_copy, g, Δt_new = Δt)
        cov_new = cov(get_u_final(ekp_copy), dims = 2)
        if det(cov_new) > cov_threshold * det(cov_init)
            accept_stepsize = true
        else
            Δt = Δt / 2
        end
    end

    return Δt
end
