module LossFunctions

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.ModelsAndData
import ..OceanTurbulenceParameterEstimation.ModelsAndData: set!

using Oceananigans
using Oceananigans.Grids: RegularRectilinearGrid
using Oceananigans.Fields: CenterField, AbstractDataField, interior

using Statistics

export
    # forward_map.jl
    ModelTimeSeries,
    model_time_series,
    
    # loss_functions.jl
    evaluate!,
    LossFunction,
    LossContainer,
    BatchedLossContainer,
    EnsembleLossContainer,
    TimeSeriesAnalysis,
    TimeAverage,
    ValueProfileAnalysis,
    GradientProfileAnalysis,
    on_grid,
    init_loss_function,
    BatchTruthData,
    BatchLossFunction

include("utils.jl")
include("time_series_analysis.jl")
include("profile_analysis.jl")
include("loss_functions.jl")
include("forward_map.jl")

end # module
