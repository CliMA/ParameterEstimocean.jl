
# Diffusion.jl

## TKE

Base.@kwdef struct TKEParameters{T} <: AbstractParameters
    Cᴰ :: T = 3.2441 # Dissipation parameter
end

## Mixing length

# Base.@kwdef struct SimpleMixingLength{T} <: AbstractParameters
#     Cᴸᵇ :: T = 3.9302
# end

# Mixing length model due to Ignacio Lopez-Gomez + Clima
Base.@kwdef struct EquilibriumMixingLength{T} <: AbstractParameters
    # Cᴸᵟ :: T = 1.0
    Cᴸʷ :: T = 1.688 # Limits to Von-Karman constant for stress-driven turbulence
    Cᴸᵇ :: T = 1.664
end

## Wall ParameterizedModels

Base.@kwdef struct PrescribedSurfaceTKEFlux{T} <: AbstractParameters
    # Cʷu★ :: T = 1.3717
    CʷwΔ :: T = 1.0
end

##  Diffusivities

Base.@kwdef struct IndependentDiffusivities{T} <: AbstractParameters
     # Cᴷu :: T = 0.0274 # Diffusivity parameter for velocity
     Cᴷc :: T = 0.0498 # Diffusivity parameter for tracers
     Cᴷe :: T = 0.0329 # Diffusivity parameter for TKE
end

Base.@kwdef struct RiDependentDiffusivities{T} <: AbstractParameters
     # Cᴷu⁻  :: T = 0.02   # Convection diffusivity parameter for velocity
     # Cᴷu⁺  :: T = 0.01   # Shift diffusivity parameter for velocity
     Cᴷc⁻  :: T = 0.04   # Convection diffusivity parameter for tracers
     Cᴷc⁺  :: T = 0.01   # Shift diffusivity parameter for tracers
     Cᴷe⁻  :: T = 0.02   # Convection diffusivity parameter for TKE
     Cᴷe⁺  :: T = 0.01   # Shift diffusivity parameter for TKE
     CᴷRiʷ :: T = 0.1    # Ri width of transition from - to +
     CᴷRiᶜ :: T = 0.1    # "Central" Ri parameter
end
