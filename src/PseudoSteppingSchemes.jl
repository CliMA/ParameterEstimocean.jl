module PseudoSteppingSchemes

import .EnsembleKalmanInversion: adaptive_step_parameters

# Default pseudo_stepping::Nothing --- it's not adaptive
adaptive_step_parameters(::Nothing, Xⁿ, Gⁿ, y, Γy, process; Δt) = step_parameters(X, G, y, Γy, process; Δt), Δt

struct ConstantConvergence{T}
    convergence_ratio :: T
end

function adaptive_step_parameters(pseudo_stepping::ConstantConvergence, Xⁿ, Gⁿ, y, Γy, process; Δt=1.0)
    convergence_ratio = pseudo_stepping.convergence_ratio

    # Test step forward
    Xⁿ⁺¹ = step_parameters(Xⁿ, Gⁿ, y, Γy, process; Δt)
    r = volume_ratio(Xⁿ⁺¹, Xⁿ)

    # "Accelerated" fixed point iteration to adjust (pseudo) Δt
    p = 1.1
    iter = 1
    while !isapprox(r, convergence_ratio, atol=0.03, rtol=0.1) && iter < 10
        Δt *= (r / convergence_ratio)^p
        Xⁿ⁺¹ = step_parameters(Xⁿ, Gⁿ, y, Γy, process; Δt)
        r = volume_ratio(Xⁿ⁺¹, Xⁿ)
        iter += 1
    end

    @info "Particles stepped adaptively with convergence ratio $r (target $convergence_ratio)"

    return Xⁿ⁺¹, Δt
end

end # module
