module LossFunctions

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.Grids
using ..OceanTurbulenceParameterEstimation.Data
using ..OceanTurbulenceParameterEstimation.Models

using Oceananigans
using Oceananigans: AbstractModel
using Oceananigans.Grids: RegularRectilinearGrid
using Oceananigans.Fields: CenterField, AbstractField, AbstractDataField, interior

using Statistics

export
    # forward_run.jl
    model_time_series,

    # loss_functions.jl
    LossFunction,
    TimeSeriesAnalysis,
    TimeAverage,
    ValueProfileAnalysis,
    GradientProfileAnalysis,
    evaluate!,
    on_grid

include("utils.jl")
include("time_analysis.jl")
include("profile_analysis.jl")
include("field_weights.jl")
include("loss_function.jl")
include("forward_run.jl")

end # module
