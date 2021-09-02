module OceanTurbulenceParameterEstimation

using Oceananigans,
      Oceananigans.Units,
      Statistics,
      StaticArrays

using Printf,
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

    # LossFunctions
    evaluate!,
    TimeSeriesAnalysis,
    TimeAverage,
    model_time_series,

    # modules
    ModelsAndData,
    CATKEVerticalDiffusivityModel,
    ParameterEstimation

include("ModelsAndData/ModelsAndData.jl")
include("CATKEVerticalDiffusivityModel/CATKEVerticalDiffusivityModel.jl")
include("LossFunctions/LossFunctions.jl")
include("ParameterEstimation/ParameterEstimation.jl")

end # module
