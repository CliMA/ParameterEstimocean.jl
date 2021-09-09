module OceanTurbulenceParameterEstimation

export

    # Grids
    OneDimensionalEnsembleGrid,

    # Observations
    OneDimensionalTimeSeries, OneDimensionalTimeSeriesBatch,

    # Models
    get_model_field, OneDimensionalEnsembleModel, ensemble_size, batch_size,
    set!, initialize_forward_run!,
    DefaultFreeParameters, get_free_parameters, FreeParameters, @free_parameters,

    # LossFunctions
    LossFunction,
    TimeSeriesAnalysis,
    TimeAverage,
    model_time_series,
    ForwardMap,
    ValueProfileAnalysis,
    GradientProfileAnalysis,

    # ParameterEstimation
    InverseProblem,
    Parameters
    
include("Grids/Grids.jl")
include("Observations/Observations.jl")
include("Models/Models.jl")
include("LossFunctions/LossFunctions.jl")
include("ParameterEstimation/ParameterEstimation.jl")

using .Grids
using .Observations
using .Models
using .LossFunctions
using .ParameterEstimation

end # module
