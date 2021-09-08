module OceanTurbulenceParameterEstimation

export

    # Grids
    ColumnEnsembleGrid,

    # Data
    OneDimensionalTimeSeries, OneDimensionalTimeSeriesBatch,

    # Models
    get_model_field, EnsembleModel, ensemble_size, batch_size,
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
    Parameters,
    relative_weight_options

include("Grids/Grids.jl")
include("Data/Data.jl")
include("Models/Models.jl")
include("LossFunctions/LossFunctions.jl")
include("ParameterEstimation/ParameterEstimation.jl")

using .Grids
using .Data
using .Models
using .LossFunctions
using .ParameterEstimation

end # module
