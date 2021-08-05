module OceanTurbulenceParameterEstimation

using Base: Float32
export
    dictify,
    set_prognostic!,

    # utils
    variance,
    max_variance,
    mean_variance,
    mean_std,
    profile_mean,
    max_gradient_variance,
    initialize_forward_run!,
    simple_safe_save,
    run_until!,

    # file_wrangling.jl
    get_iterations,
    get_times,
    get_data,
    get_parameter,
    get_grid_params,

    # models_and_data.jl
    TruthData,
    ParameterizedModel,

    # free_parameters.jl
    DefaultFreeParameters,
    get_free_parameters,
    FreeParameters,
    @free_parameters,

    # free_parameters.jl and set_fields.jl
    set!,

    # visualization.jl
    defaultcolors,
    removespine,
    removespines,
    plot_data!,
    format_axs!,
    visualize_realizations,
    visualize_loss_function,
    visualize_markov_chain!,
    plot_loss_function,

    # loss_functions.jl
    evaluate!,
    analyze_weighted_profile_discrepancy,
    VarianceWeights,
    LossFunction,
    TimeSeriesAnalysis,
    TimeAverage,
    ValueProfileAnalysis,
    GradientProfileAnalysis,
    on_grid,
    init_negative_log_likelihood,

    # forward_map.jl
    ParameterizedModelTimeSeries,
    model_time_series,

    # modules
    TKEMassFluxModel,
    ParameterEstimation

using Oceananigans,
      Oceananigans.Units,
      Statistics,
      StaticArrays,
      Dao

using Plots,
      Printf,
      FileIO,
      JLD2,
      OffsetArrays

import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Fields: interpolate

using Oceananigans: AbstractModel
using Oceananigans.Fields: CenterField, AbstractDataField
using Oceananigans.Grids: Face, Center, AbstractGrid
using Oceananigans.TurbulenceClosures: TKEBasedVerticalDiffusivity

using JLD2

using PyPlot, PyCall

import Base: length

abstract type FreeParameters{N, T} <: FieldVector{N, T} end

function Base.similar(p::FreeParameters{N, T}) where {N, T}
    P = typeof(p).name.wrapper
    return P((zero(T) for i=1:N)...)
end

Base.show(io::IO, p::FreeParameters) = print(io, "$(typeof(p)):", '\n',
                                             @sprintf("% 24s: ", "parameter names"),
                                             (@sprintf("%-8s", n) for n in propertynames(p))..., '\n',
                                             @sprintf("% 24s: ", "values"),
                                             (@sprintf("%-8.4f", pᵢ) for pᵢ in p)...)

dictify(p) = Dict((k, getproperty(p, k)) for k in propertynames(p))

include("file_wrangling.jl")
include("set_fields.jl")
include("models_and_data.jl")
include("free_parameters.jl")
include("utils.jl")
include("visualization.jl")
include("loss_functions.jl")
include("forward_map.jl")

include("TKEMassFluxModel/TKEMassFluxModel.jl")
include("ParameterEstimation/ParameterEstimation.jl")

end # module
