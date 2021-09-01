module LossFunctions

using Oceananigans: AbstractModel
using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.ModelsAndData
import ..OceanTurbulenceParameterEstimation.ModelsAndData: set!, get_model_field

using Oceananigans
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
include("loss_functions.jl")
include("forward_run.jl")

end # module
