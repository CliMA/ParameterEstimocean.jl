module ParameterEstimation

using Distributions: delta
using FileIO, Optim, Random, Dao,
        Statistics, Distributions, LinearAlgebra,
        PyPlot, Optim, Printf, LaTeXStrings, CairoMakie
using Dao: AdaptiveAlgebraicSchedule

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.ModelsAndData
using ..OceanTurbulenceParameterEstimation.CATKEVerticalDiffusivityModel
using ..OceanTurbulenceParameterEstimation.LossFunctions

using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

import ..OceanTurbulenceParameterEstimation.ModelsAndData: set!

using Oceananigans.Fields: interior
using CairoMakie: Figure

export
       Parameters,
       DataSet,
       CalibrationExperiment,
       validation_loss_reduction,
       relative_weight_options,
       set!,

       # catke_vertical_diffusivity_model_setup.jl
       get_loss,
       dataset,
       ensemble_dataset,

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

       # visualization.jl
       visualize_realizations,
       visualize_and_save

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

struct DataSet{LD, DB, PM, RW, LC, FP}
        LESdata::LD
        data_batch::DB
        model::PM
        relative_weights::RW # field weights
        loss::LC
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

include("utils.jl")
include("catke_vertical_diffusivity_model_setup.jl")
include("calibration_algorithms/calibration_algorithms.jl")
include("visualization.jl")

end # module
