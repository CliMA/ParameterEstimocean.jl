module OceanTurbulenceParameterEstimation

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
    simple_safe_save,

    # file_wrangling.jl
    get_iterations,
    get_times,
    get_data,
    get_parameter,
    get_grid_params,

    # models_and_data.jl
    TruthData,
    ParameterizedModel,
    run_until!,
    initialize_forward_run!,

    # free_parameters.jl
    DefaultFreeParameters,
    get_free_parameters,
    FreeParameters,
    @free_parameters,

    # free_parameters.jl / 
    # set_fields.jl / 
    # single_column_utils.jl / 
    # many_columns_utils.jl
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
    BatchLossFunction,

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

using Oceananigans: AbstractModel, AbstractEddyViscosityClosure
using Oceananigans.Fields: CenterField, AbstractDataField
using Oceananigans.Grids: Face, Center, AbstractGrid
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, AbstractTurbulenceClosure

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

# Temporary
include("to_import.jl")

include("file_wrangling.jl")
include("set_fields.jl")
include("models_and_data.jl")
include("free_parameters.jl")
include("loss_function_utils.jl")
include("many_columns_utils.jl")
include("loss_functions.jl")
include("single_column_utils.jl")
include("utils.jl")
include("visualization.jl")
include("forward_map.jl")

include("TKEMassFluxModel/TKEMassFluxModel.jl")
include("ParameterEstimation/ParameterEstimation.jl")

end # module
