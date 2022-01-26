# In this example, we use EKI to tune the closure parameters of a HydrostaticFreeSurfaceModel 
# with a CATKEBasedVerticalDiffusivity closure in order to align the predictions of the model 
# to those of a high-resolution LES data generated in LESbrary.jl. Here `predictions` refers to the
# 1-D profiles of temperature, velocity, and turbulent kinetic energy horizontally averaged over a
# 3-D physical domain.

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using LinearAlgebra, Distributions, JLD2, DataDeps
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using OceanTurbulenceParameterEstimation

include("utils/lesbrary_paths.jl")
include("utils/one_dimensional_ensemble_model.jl")
include("utils/parameters.jl")
include("utils/visualize_profile_predictions.jl")

# two_day_suite_dir = "/Users/gregorywagner/Projects/OceanTurbulenceParameterEstimation/data/2DaySuite"
# four_day_suite_dir = "/Users/gregorywagner/Projects/OceanTurbulenceParameterEstimation/data/4DaySuite"
# six_day_suite_dir = "/Users/gregorywagner/Projects/OceanTurbulenceParameterEstimation/data/6DaySuite"

# two_day_suite = TwoDaySuite(two_day_suite_dir)
# four_day_suite = FourDaySuite(four_day_suite_dir)
# six_day_suite = SixDaySuite(six_day_suite_dir)

# calibration = InverseProblem(two_day_suite, parameters; relative_weights = relative_weight_options["all_but_e"],
#                              architecture = GPU(), ensemble_size = 10, Δt = 30.0)

# validation = InverseProblem(four_day_suite, calibration; Nz = 32);

#####
##### Set up ensemble model
#####

lesbrary_directory = "/Users/adelinehillier/Desktop/dev/"
lesbrary_directory = "/home/ahillier/home/"

observations = TwoDaySuite(lesbrary_directory; first_iteration = 13, last_iteration = nothing, normalize = ZScore, Nz = 128)

parameter_set = CATKEParametersRiDependent
closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

ensemble_model = OneDimensionalEnsembleModel(observations;
    architecture = GPU(),
    ensemble_size = 20,
    closure = closure
)

ensemble_simulation = Simulation(ensemble_model; Δt = 10seconds, stop_time = 2days)

#####
##### Build free parameters
#####

build_prior(name) = ConstrainedNormal(0.0, 1.0, bounds(name) .* 0.5...)
free_parameters = FreeParameters(named_tuple_map(names(parameter_set), build_prior))

#####
##### Build the Inverse Problem
#####

# Specify an output map that tracks 3 uniformly spaced time steps, ignoring the initial condition
track_times = Int.(floor.(range(1, stop = length(observations[1].times), length = 3)))
calibration = InverseProblem(observations, ensemble_simulation, free_parameters; output_map = ConcatenatedOutputMap(track_times))

# #####
# ##### Calibrate
# #####

iterations = 5
eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)
params = iterate!(eki; iterations = iterations)

directory = "calibrate_catke_to_lesbrary/"
isdir(directory) || mkpath(directory)

visualize!(calibration, params;
    field_names = [:u, :v, :b, :e],
    directory = @__DIR__,
    filename = "perfect_model_visual_calibrated.png"
)
@show params
