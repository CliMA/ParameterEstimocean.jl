module TKEMassFluxModel

export parameter_latex_guide

using ..OceanTurbulenceParameterEstimation
using LaTeXStrings

using Oceananigans.BoundaryConditions
using Oceananigans.BuoyancyModels: BuoyancyTracer
using Oceananigans.Coriolis: FPlane
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Suppressor

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity,
                MixingLength, SurfaceTKEFlux, VerticallyImplicitTimeDiscretization

# CATKEVerticalDiffusivity{TD, A, B, C, D}(a::A, b::B, c::C, d::D) where {TD, A, B, C, D} = CATKEVerticalDiffusivity{TD}(a, b, c, d)

export TKEParametersRiDependent,
       TKEParametersRiDependentConvectiveAdjustment,
       TKEParametersRiIndependent,
       TKEParametersRiIndependentConvectiveAdjustment,

       custom_defaults,
       parameter_latex_guide,
       free_parameter_options,
       get_bounds_and_variance,
       parameter_specific_kwargs

include("tke_mass_flux_parameters.jl")
include("tke_mass_flux_model_single_column.jl")
include("tke_mass_flux_model_many_columns.jl")

end # module
