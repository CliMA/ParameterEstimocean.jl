module CATKEVerticalDiffusivityModel

export parameter_latex_guide

using ..Grids
using ..Observations
using ..Models

using Oceananigans
using Oceananigans: AbstractModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity,
                MixingLength, SurfaceTKEFlux, VerticallyImplicitTimeDiscretization

using Suppressor, LaTeXStrings

export CATKEParametersRiDependent,
       CATKEParametersRiDependentConvectiveAdjustment,
       CATKEParametersRiIndependent,
       CATKEParametersRiIndependentConvectiveAdjustment,

       custom_defaults,
       parameter_latex_guide,
       free_parameter_options,
       get_bounds_and_variance,
       parameter_specific_kwargs

include("catke_vertical_diffusivity_parameters.jl")
include("catke_vertical_diffusivity_model.jl")

end # module
