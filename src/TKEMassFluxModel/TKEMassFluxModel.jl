module TKEMassFluxModel

export parameter_latex_guide

using ..OceanTurbulenceParameterEstimation
using LaTeXStrings
using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: TKEBasedVerticalDiffusivity, RiDependentDiffusivityScaling, VerticallyImplicitTimeDiscretization

using Oceananigans.BoundaryConditions
using Oceananigans.BuoyancyModels: BuoyancyTracer
using Oceananigans.Coriolis: FPlane
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Suppressor

export TKEParametersRiDependent,
       TKEFreeConvection,
       TKEBCParameters,

       custom_defaults,
       parameter_latex_guide,
       free_parameter_options,
       get_bounds_and_variance,
       parameter_specific_kwargs

include("tke_mass_flux_parameters.jl")
include("tke_mass_flux_models.jl")

end # module
