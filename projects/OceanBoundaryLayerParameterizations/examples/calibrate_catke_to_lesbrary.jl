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
using OceanBoundaryLayerParameterizations
using OceanLearning
using OceanLearning.Parameters: closure_with_parameters

Nz = 32
Nensemble = 20
architecture = CPU()
Δt = 10.0

two_day_suite = TwoDaySuite(; Nz)
four_day_suite = FourDaySuite(; Nz)
six_day_suite = SixDaySuite(; Nz)

#####
##### Set up ensemble model
#####

observations = two_day_suite

parameter_set = CATKEParametersRiDependent
closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

#####
##### Build free parameters
#####

build_prior(name) = ScaledLogitNormal(bounds=bounds(name).*0.5)
free_parameters = FreeParameters(named_tuple_map(parameter_set.names, build_prior))

#####
##### Build the Inverse Problem
#####

track_times = Int.(floor.(range(1, stop = length(observations[1].times), length = 3)))
output_map = ConcatenatedOutputMap(track_times)

function build_inverse_problem(Nensemble)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, closure, Δt)
    calibration = InverseProblem(observations, simulation, free_parameters; output_map)
    return calibration
end

calibration = build_inverse_problem(Nensemble)


y = observation_map(calibration);
θ = named_tuple_map(parameter_set.names, default)
G = forward_map(calibration, [θ])
zc = [mapslices(norm, G .- y, dims = 1)...]

y = observation_map(calibration);
θ = 
G = forward_map(calibration, [θ])
zc = [mapslices(norm, G .- y, dims = 2)...]

#####
##### Calibrate
#####

iterations = 2
eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2,
                                        resampler = Resampler(acceptable_failure_fraction=0.5, only_failed_particles=true))
params = iterate!(eki; iterations = iterations)

directory = "calibrate_catke_to_lesbrary/"
isdir(directory) || mkpath(directory)

visualize!(calibration, params;
    field_names = [:u, :v, :b, :e],
    directory = @__DIR__,
    filename = "perfect_model_visual_calibrated.png"
)
@show params
