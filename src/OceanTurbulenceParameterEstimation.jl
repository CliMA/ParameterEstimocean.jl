module OceanTurbulenceParameterEstimation

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

using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

export
    ColumnEnsembleSize,
    
    # ModelsAndData
    TruthData,
    run_until!,
    initialize_forward_run!,
    DefaultFreeParameters,
    get_free_parameters,
    FreeParameters,
    @free_parameters,
    set!,

    # ModelsAndData/LESbrary_paths.jl
    LESbrary,
    TwoDaySuite,
    FourDaySuite,
    SixDaySuite,
    GeneralStrat,

    # ParameterEstimation
    visualize_realizations,

    # LossFunctions
    evaluate!,
    LossFunction,
    LossContainer,
    BatchedLossContainer,
    EnsembleLossContainer,
    TimeSeriesAnalysis,
    TimeAverage,
    init_loss_function,
    BatchTruthData,
    BatchLossFunction,
    ModelTimeSeries,
    model_time_series,

    # modules
    ModelsAndData,
    CATKEVerticalDiffusivityModel,
    ParameterEstimation

# Temporary
include("to_import.jl")

include("ModelsAndData/ModelsAndData.jl")
include("CATKEVerticalDiffusivityModel/CATKEVerticalDiffusivityModel.jl")
include("LossFunctions/LossFunctions.jl")
include("ParameterEstimation/ParameterEstimation.jl")

end # module
