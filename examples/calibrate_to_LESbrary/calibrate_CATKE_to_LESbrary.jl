pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using Plots, LinearAlgebra, Distributions, JLD2
using Oceananigans.Units
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
##
##

directory = "/Users/adelinehillier/Desktop/dev/"

observations = TwoDaySuite(directory; first_iteration = 13, last_iteration = nothing, normalize = ZScore, Nz = 128)

parameter_set = CATKEParametersRiDependent
closure = closure_with_parameter_set(CATKEVerticalDiffusivity(Float64;), parameter_set)

ensemble_model = OneDimensionalEnsembleModel(observations;
    architecture = CPU(),
    ensemble_size = 20,
    closure = closure
)

ensemble_simulation = Simulation(ensemble_model; Δt = 10seconds, stop_time = 2days)

pop!(ensemble_simulation.diagnostics, :nan_checker)

#####
##### Build free parameters
#####

free_parameter_names = keys(parameter_set.defaults)
free_parameter_means = collect(values(parameter_set.defaults))
priors = NamedTuple(pname => ConstrainedNormal(0.0, 1.0, bounds(pname) .* 0.5...) for pname in free_parameter_names)

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters);

# #####
# ##### Calibrate
# #####

iterations = 5
# eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)
# params = iterate!(eki; iterations = iterations)

# visualize!(calibration, params;
#     field_names = [:u, :v, :b, :e],
#     directory = @__DIR__,
#     filename = "perfect_model_visual_calibrated.png"
# )
# @show params
