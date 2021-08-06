module ParameterEstimation

using JLD2, FileIO, Printf, PyPlot, Optim, Random,
        Statistics, Distributions, LinearAlgebra, OrderedCollections
using PyPlot, Optim, OrderedCollections
using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.TKEMassFluxModel
using Dao
using Dao: AdaptiveAlgebraicSchedule

using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
using Oceananigans.TurbulenceClosures: RiDependentDiffusivityScaling, VerticallyImplicitTimeDiscretization
using Oceananigans.Grids: Flat, Bounded, Periodic, RegularRectilinearGrid

import ..OceanTurbulenceParameterEstimation: set!

export
       Parameters,
       DataSet,
       CalibrationExperiment,
       validation_loss_reduction,
       relative_weight_options,
       set!,

       # tke_utils.jl
       get_loss,
       init_tke_calibration,
       dataset,

       # grids.jl
       ZGrid,
       XYZGrid,

       # LESbrary_paths.jl
       LESbrary,
       TwoDaySuite,
       FourDaySuite,
       SixDaySuite,
       GeneralStrat,

       # visuals.jl
       visualize_and_save,

       # utils.jl
       open_output_file,
       writeout,

       # algorithms.jl
       eki_unidimensional,
       eki_multidimensional,
       simulated_annealing,
       nelder_mead,
       l_bfgs,
       random_plugin,
       gradient_descent,

       # TKECalibration2021
       custom_tke_calibration,

       # OceanTurbulenceParameterEstimation
       visualize_realizations,
       FreeParameters,
       @free_parameters,
       set!

relative_weight_options = Dict(
                "all_e"     => Dict(:b => 0.0, :u => 0.0, :v => 0.0, :e => 1.0),
                "all_T"     => Dict(:b => 1.0, :u => 0.0, :v => 0.0, :e => 0.0),
                "uniform"   => Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0),
                "all_but_e" => Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 0.0),
                "all_uv"    => Dict(:b => 0.0, :u => 1.0, :v => 1.0, :e => 0.0),
                "mostly_T"  => Dict(:b => 1.0, :u => 0.5, :v => 0.5, :e => 0.0)
)

Base.@kwdef struct Parameters{T <: UnionAll}
    RelevantParameters::T
    ParametersToOptimize::T
end

struct DataSet{LD, RW, NLL, FP}
        LESdata::LD
        relative_weights::RW # field weights
        loss::NLL
        default_parameters::FP
end

struct CalibrationExperiment{DS, PP, FP}
        calibration::DS
        validation::DS
        parameters::PP
        default_parameters::FP
end

function CalibrationExperiment(calibration, validation, parameters)
    CalibrationExperiment(calibration, validation, parameters, calibration.default_parameters)
end

function validation_loss_reduction(ce::CalibrationExperiment, parameters::FreeParameters)
    validation_loss = ce.validation.loss(parameters)
    calibration_loss = ce.calibration.loss(parameters)

    default_validation_loss = ce.validation.loss(ce.default_parameters)
    default_calibration_loss = ce.calibration.loss(ce.default_parameters)

    validation_loss_reduction = validation_loss/default_validation_loss
    println("Parameters: $([parameters...])")
    println("Validation loss reduction: $(validation_loss_reduction)")
    println("Training loss reduction: $(calibration_loss/default_calibration_loss)")

    return validation_loss_reduction
end

include("LESbrary_paths.jl")
include("grids.jl")
include("utils.jl")
include("tke_mass_flux_model_setup.jl")
include("calibration_algorithms/calibration_algorithms.jl")
include("visuals.jl")

end # module
