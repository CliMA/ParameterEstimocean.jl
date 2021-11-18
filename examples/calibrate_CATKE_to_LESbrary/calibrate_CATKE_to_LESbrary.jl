pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using Plots, LinearAlgebra, Distributions, JLD2
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using OceanTurbulenceParameterEstimation

include("lesbrary_paths.jl")
include("one_dimensional_ensemble_model.jl")
include("parameters.jl")
include("visualize_profile_predictions.jl")

#####
##### Set up ensemble model
#####

## NEED TO IMPLEMENT COARSE-GRAINING

directory = "/Users/adelinehillier/Desktop/dev/"

observations = TwoDaySuite(directory; first_iteration = 13, last_iteration = nothing, normalize = ZScore, Nz = 128)

parameter_set = CATKEParametersRiDependent
closure = closure_with_parameter_set(CATKEVerticalDiffusivity(Float64;), parameter_set)

ensemble_model = OneDimensionalEnsembleModel(observations;
    architecture = CPU(),
    ensemble_size = 50,
    closure = closure
)

ensemble_simulation = Simulation(ensemble_model; Δt = 10seconds, stop_time = 2days)

pop!(ensemble_simulation.diagnostics, :nan_checker)
