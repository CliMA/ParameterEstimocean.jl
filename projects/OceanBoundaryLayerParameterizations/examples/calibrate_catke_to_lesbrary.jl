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
using OceanBoundaryLayerParameterizations

two_day_suite = TwoDaySuite()
four_day_suite = FourDaySuite()
six_day_suite = SixDaySuite()

#####
##### Set up ensemble model
#####

observations = TwoDaySuite(lesbrary_directory;)

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

build_prior(name) = ScaledLogitNormal(bounds=bounds(name).*0.5)
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
