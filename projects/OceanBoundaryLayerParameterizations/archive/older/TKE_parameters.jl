

# Parameters for TKEMassFlux
# Always have
# 	Cᴰ : “dissipation parameter” in TKE equation
# 	Cᴸʷ : mixing length parameter, distance to wall
# 	Cᴸᵇ : mixing length parameter, decrease in stable stratification
# 	CʷwΔ: surface flux of TKE due to destabilizing buoyancy flux

# Then with
# 	eddy_diffusivities = IndependentDiffusivities()
# 	Cᴷc: Diffusivity magnitude for tracers
# 	Cᴷe: Diffusivity magnitude for TKE

# With eddy_diffusivities = RiDependentDiffusivities()
# 	CᴷRiʷ, CᴷRiᶜ: Ri width of transition from - to + constant and central Ri
# 	Cᴷc⁻, Cᴷc⁺ : lower and upper diffusivity constants for tracers
# 	Cᴷe⁻, Cᴷe⁺ : lower and upper diffusivity constants for TKE
	
# With convective_adjustment = TKEMassFlux.FluxProportionalConvectiveAdjustment()
# 	Cᴬ : Magnitude of the convective adjustment diffusivity

# I suggest depending on OceanTurb #glw/convective-adjustment-defaults and starting with
eddy_diffusivities = RiDependentDiffusivities()
convective_adjustment = TKEMassFlux.FluxProportionalConvectiveAdjustment()

# Create a FreeParameters struct in `OceanTurbulenceParameterEstimation` (or in https://github.com/glwagner/OceanTurbulenceParameterEstimation/blob/master/projects/tke_with_convection/setup.jl) with

# Another possibility is to use IndependentDiffusivities…
eddy_diffusivities = IndependentDiffusivities()
convective_adjustment = TKEMassFlux.FluxProportionalConvectiveAdjustment()
# with
@free_parameters(ConvectiveAdjustmentIndependentDiffusivitesTKEParameters,
                 Cᴷc, Cᴷe,
                 Cᴰ, Cᴸᵇ, CʷwΔ, Cᴬ)



# Diffusion.jl

## TKE

Base.@kwdef struct TKEParameters{T} <: AbstractParameters
    Cᴰ :: T = 3.2441 # Dissipation parameter
end


## Nonlocal flux

Base.@kwdef struct CounterGradientFlux{T} <: AbstractParameters
	Cᴺ :: T = 1.0 # Mass flux proportionality constant
end

abstract type AbstractDiagnosticPlumeModel <: AbstractParameters end

Base.@kwdef struct WitekDiagnosticPlumeModel{T} <: AbstractDiagnosticPlumeModel
     Ca :: T = 0.1
    Cbw :: T = 2.86
     Ce :: T = 0.1
    Cew :: T = 0.572
     CQ :: T = 1.0
end


## Mixing length

Base.@kwdef struct SimpleMixingLength{T} <: AbstractParameters
    Cᴸᵇ :: T = 3.9302
end

# Mixing length model due to Ignacio Lopez-Gomez + Clima
Base.@kwdef struct EquilibriumMixingLength{T} <: AbstractParameters
    Cᴸᵟ :: T = 1.0
    Cᴸʷ :: T = 1.688 # Limits to Von-Karman constant for stress-driven turbulence
    Cᴸᵇ :: T = 1.664
end

## Wall Models

Base.@kwdef struct PrescribedNearWallTKE{T} <: AbstractParameters
    Cʷu★ :: T = 3.75
end

Base.@kwdef struct PrescribedSurfaceTKEValue{T} <: AbstractParameters
    Cʷu★ :: T = 3.75
end

Base.@kwdef struct PrescribedSurfaceTKEFlux{T} <: AbstractParameters
    Cʷu★ :: T = 1.3717
    CʷwΔ :: T = 1.0
end

##  Diffusivities

Base.@kwdef struct BackgroundDiffusivities{T} <: AbstractParameters
    KU₀ :: T = 1e-6 # Background viscosity for momentum [m s⁻²]
    KC₀ :: T = 1e-6 # Background diffusivity for tracers and TKE [m s⁻²]
end

Base.@kwdef struct SinglePrandtlDiffusivities{T} <: AbstractParameters
     Cᴷu :: T = 0.1   # Diffusivity parameter for velocity
    CᴷPr :: T = 0.74  # Constant Prandtl number for tracers and TKE
end

Base.@kwdef struct IndependentDiffusivities{T} <: AbstractParameters
     Cᴷu :: T = 0.0274 # Diffusivity parameter for velocity
     Cᴷc :: T = 0.0498 # Diffusivity parameter for tracers
     Cᴷe :: T = 0.0329 # Diffusivity parameter for TKE
end

Base.@kwdef struct RiDependentDiffusivities{T} <: AbstractParameters
     Cᴷu⁻  :: T = 0.02   # Convection diffusivity parameter for velocity
     Cᴷu⁺  :: T = 0.01   # Shift diffusivity parameter for velocity
     Cᴷc⁻  :: T = 0.04   # Convection diffusivity parameter for tracers
     Cᴷc⁺  :: T = 0.01   # Shift diffusivity parameter for tracers
     Cᴷe⁻  :: T = 0.02   # Convection diffusivity parameter for TKE
     Cᴷe⁺  :: T = 0.01   # Shift diffusivity parameter for TKE
     CᴷRiʷ :: T = 0.1    # Ri width parameter
     CᴷRiᶜ :: T = 0.1    # "Central" Ri parameter
end
