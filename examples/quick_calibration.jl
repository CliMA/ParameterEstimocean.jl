using OceanTurbulenceParameterEstimation
using Oceananigans
using Distributions

#
# Define observations, or "truth data".
#
# For observational time-series, we specify
# the time-range that the observations are defined, which may be different from the
# data contained in a file.
#
# If the observations are one-dimensional, then forward maps are constructed that use
# a single 3D Oceananigans model to evaluate an ensemble of column models.
#
# If the observations are two-dimensional, then forward maps are constructed that use
# a single 3D Oceananigans model to evaluate an ensemble of "slice" 2D models.
#

free_convection_path = "/path/to/data"
free_convection_observation = OneDimensionalTimeSeries(free_convection_path, normalization=Variance(), field_names=(:u, :v, :b, :e), time_range=range(2hours, stop=2days, step=4hours))

# normalize(::Variance, field) =

# Some other possibilities

# 1. Construct a batch of observations:
# wind_stress_observation = OneDimensionalTimeSeries(wind_stress_path, time_range=range(2hours, stop=2days, step=4hours))

file_paths = [free_convection_path, wind_stress_path]
observations = [OneDimensionalTimeSeries(path, field_names=(:u, :v, :b, :e)) for path in file_paths]

const OneDimensionalTimeSeriesBatch = Vector{<:OneDimensionalTimeSeries}
const TimeSeriesBatch = Vector{<:AbstractTruthData}

struct OneDimensionalTimeSeries <: AbstractTruthData
    field_timeseries
    times
    iterations
    forward_map_time_range
    meta_data
end


stressed = model.velocities.u.boundary_conditions.top.condition == 0
rotating = model.coriolis.f == 0

struct TruthData{F, G, C, D, UU, VV, BΘ, EE, TT, NN, TG, RF}
   boundary_conditions :: F
                  grid :: G
             constants :: C
         diffusivities :: D
                     u :: UU
                     v :: VV
                     b :: BΘ
                     e :: EE
                     t :: TT
                  name :: NN
               targets :: TG
       relevant_fields :: RF
end


# 2. Construct two-dimensional observations at a single snapshot:
# observation = TwoDimensionalSnapshot(data_path, iteration=LastIteration())

# EKI solves
#
# y = G(θ) + η
#
# where G(θ) is the forward map and y is the observations.

@free_parameters StabilityFnParameters CᴷRiʷ CᴷRiᶜ Cᴷu⁻ Cᴷuʳ Cᴷc⁻ Cᴷcʳ Cᴷe⁻ Cᴷeʳ
@free_parameters StabilityFnParameters CᴷRiʷ CᴷRiᶜ Cᴷuʳ Cᴷc⁻ Cᴷcʳ Cᴷe⁻ Cᴷeʳ

stability_fn_parameters_priors = StabilityFnParameters(
    CᴷRiʷ = Normal(0.25, 0.05) |> logify,
    CᴷRiᶜ = Normal(0.25, 0.5) |> logify,
    Cᴷu⁻ = Uniform(0.0, 10.0),
    Cᴷuʳ = Uniform(0.0, 10.0),
    Cᴷc⁻ = Uniform(0.0, 10.0),
    Cᴷcʳ = Uniform(0.0, 10.0),
    Cᴷe⁻ = Uniform(0.0, 10.0),
    Cᴷeʳ = Uniform(0.0, 10.0)
)

free_parameters = FreeParameters(parameters = StabilityFnParameters,
                                 prior = stability_fn_parameters_priors)

free_parameters = FreeParameters(parameters = BoundaryLayerMesoscaleCombinedParameters)

# Tricky question: where do we build the model?
#
# 1. Build the model in ForwardMap, dispatching on the type of observations
# 2. Just ask users to "pull up their pants" and build the model in their scripts.

# Define the forward map.
#
# With forward_map = ForwardMap(), G(θ) outputs a concatenated vector of all fields
# at all times.
#
# With forward_map = TimeAveragedForwardMap(If forwardloss_function = TimeAveragedLossFunction() (for example), then G(θ) outputs a _scalar_,
# representing some time-averaged measurement of discrepency between model and data.
# 
# Questions:
#
# * How do we handle NaNs

struct InverseProblem
    observations
    simulation
    transformation
    parameters
end

simple_calibration_problem = InverseProblem(simulation, observation, transformation=nothing)
time_averaged_calibration_problem = InverseProblem(simulation, observation, transformation=AccumulatedTimeAverage())

calibrate!(calibration_problem)