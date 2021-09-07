# In this example, we use EKI to tune the closure parameters of a HydrostaticFreeSurfaceModel 
# with a CATKEBasedVerticalDiffusivity closure in order to align the predictions of the model 
# to those of a high-resolution LES data generated in LESbrary.jl. Here `predictions` refers to the
# 1-D profiles of temperature, velocity, and turbulent kinetic energy horizontally averaged over a
# 3-D physical domain.

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "projects", "OceanBoundaryLayerParameterizations", "src"))

using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.Models.CATKEVerticalDiffusivityModel

using OceanBoundaryLayerParameterizations

# CATKE parameters involved in setting field diffusivities 
@free_parameters StabilityFnParameters CᴷRiʷ CᴷRiᶜ Cᴷu⁻ Cᴷuʳ Cᴷc⁻ Cᴷcʳ Cᴷe⁻ Cᴷeʳ

parameters = Parameters(
    RelevantParameters = CATKEParametersRiDependent,  # Parameters that are used in CATKE
    ParametersToOptimize = StabilityFnParameters    # Subset of RelevantParameters that we want to optimize
)

two_day_suite_dir = "/Users/gregorywagner/Projects/OceanTurbulenceParameterEstimation/data/2DaySuite"
four_day_suite_dir = "/Users/gregorywagner/Projects/OceanTurbulenceParameterEstimation/data/4DaySuite"
six_day_suite_dir = "/Users/gregorywagner/Projects/OceanTurbulenceParameterEstimation/data/6DaySuite"

two_day_suite = TwoDaySuite(two_day_suite_dir)
four_day_suite = FourDaySuite(four_day_suite_dir)
six_day_suite = SixDaySuite(six_day_suite_dir)

# InverseProblem represents the model, data, loss function, and parameters
calibration = InverseProblem(two_day_suite, # "Truth data" for model calibration
                             parameters;   # Model parameters 
                             # Loss function parameters
                             relative_weights = relative_weight_options["all_but_e"],
                             # Model (hyper)parameters
                             ensemble_size = 10,
                             Nz = 16,
                             Δt = 30.0)

validation = InverseProblem(four_day_suite, 
                            parameters;
                            relative_weights = relative_weight_options["all_but_e"],
                            ensemble_size = 10,
                            Nz = 64,
                            Δt = 10.0);

# Loss on default parameters
#l0 = calibration()

# Example parameters
θ = calibration.default_parameters

# Loss on parameters θ.
# θ can be 
#   1. a vector
#   2. a FreeParameters object
#   3. a vector of parameter vectors (one for each ensemble member)
#   4. or a vector of FreeParameter objects (one for each ensemble member)
# If (1) or (2), the ensemble members are redundant and the loss is computed for just the one parameter set.
#lθ = calibration(θ)

# Output files/figures
directory = joinpath(pwd(), "quick_calibrate")

# Run the model forward and store the solution
output = model_time_series(calibration, θ)

# Run the model forward with parameters θ and visualize the solution compared to the truth
visualize_predictions(calibration, θ; filename = "visualize_predictions_default_parameters.png")

#=
function visualize_predictions(output::ModelTimeSeries; filename)
    # code
    return nothing
end

visualize_predictions(calibration::InverseProblem, θ; kwargs...) = visualize_predictions(model_time_series(calibration, θ)...; kwargs...)
=#

@info [output[1].b[2].data...]
initialize_forward_run!(calibration.model, calibration.data_batch, parameters, calibration.loss.first_targets)

# Runs `visualize_predictions` and records a summary of the calibration results in a `result.txt` file.
image_dir = joinpath(@__DIR__, "quick_calibrate")
mkpath(image_dir)
visualize_and_save(calibration, validation, θ, directory)

# Use EKI to calibrate the model parameters
# eki(calibration, initial_parameters;
#     noise_level = 10^(-2.0),
#     N_iter = 15,
#     stds_within_bounds = 0.6,
#     informed_priors = false)

# plot_prior_variance_and_obs_noise_level(calibration, validation, initial_parameters, directory; vrange=0.40:0.025:0.90, nlrange=-2.5:0.1:0.5)
