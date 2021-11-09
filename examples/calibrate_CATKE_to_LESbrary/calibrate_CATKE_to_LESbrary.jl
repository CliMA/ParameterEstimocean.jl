pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using Plots, LinearAlgebra, Distributions, JLD2
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using OceanTurbulenceParameterEstimation

include("lesbrary_paths.jl")
include("parameters.jl")
include("visualize_profile_predictions.jl")



#####
##### Set up ensemble model
#####

ensemble_model = OneDimensionalEnsembleModel(observations; 
                       architecture = CPU(), 
                       ensemble_size = 50, 
                       closure = closure
                      )

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

pop!(ensemble_simulation.diagnostics, :nan_checker)
