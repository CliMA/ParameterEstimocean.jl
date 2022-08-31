module PseudoSteppingSchemes

using Statistics
using LinearAlgebra
using Printf

export ConstantConvergence

using ..EnsembleKalmanInversions: step_parameters
import ..EnsembleKalmanInversions: adaptive_step_parameters

struct ConstantConvergence{T}
    convergence_ratio :: T
end

function volume_ratio(Xⁿ⁺¹, Xⁿ)
    Vⁿ⁺¹ = det(cov(Xⁿ⁺¹, dims=2))
    Vⁿ   = det(cov(Xⁿ,   dims=2))
    return Vⁿ⁺¹ / Vⁿ
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

    # A nice message
    intro_str       = "Adaptive step found for ConstantConvergence pseudostepping."
    convergence_str = @sprintf("      ├─ convergence ratio: %.6f (target: %.2f)", r, convergence_ratio)
    time_step_str   = @sprintf("      └─ psuedo time step: %.3e", Δt)
    @info string(intro_str, '\n', convergence_str, '\n', time_step_str)

    return Xⁿ⁺¹, Δt
end

end # module
