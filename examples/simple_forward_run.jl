# In this example, we use EKI to tune the closure parameters of a HydrostaticFreeSurfaceModel 
# with a CATKEBasedVerticalDiffusivity closure in order to align the predictions of the model 
# to those of a high-resolution LES data generated in LESbrary.jl. Here `predictions` refers to the
# 1-D profiles of temperature, velocity, and turbulent kinetic energy horizontally averaged over a
# 3-D physical domain.

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "projects", "OceanBoundaryLayerParameterizations", "src"))

using Oceananigans
using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.Models.CATKEVerticalDiffusivityModel
using OceanBoundaryLayerParameterizations

# CATKE parameters involved in setting field diffusivities 
@free_parameters StabilityFnParameters CᴷRiʷ CᴷRiᶜ Cᴷu⁻ Cᴷuʳ Cᴷc⁻ Cᴷcʳ Cᴷe⁻ Cᴷeʳ

parameters = Parameters(
    RelevantParameters = CATKEParametersRiDependent,  # Parameters that are used in CATKE
    ParametersToOptimize = StabilityFnParameters    # Subset of RelevantParameters that we want to optimize
)

two_day_suite_dir = "/home/greg/Projects/OceanTurbulenceParameterEstimation/data/2DaySuite"
two_day_suite = TwoDaySuite(two_day_suite_dir)

# InverseProblem represents the model, data, loss function, and parameters
calibration = InverseProblem(two_day_suite, # "Truth data" for model calibration
                             parameters;   # Model parameters 
                             # Loss function parameters
                             relative_weights = relative_weight_options["all_but_e"],
                             # Model (hyper)parameters
                             architecture = GPU(),
                             ensemble_size = 10,
                             Nz = 16,
                             Δt = 1.0)

# Run the model forward and store the solution
output = model_time_series(calibration, calibration.default_parameters)
