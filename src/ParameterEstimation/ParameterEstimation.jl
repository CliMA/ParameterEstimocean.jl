module ParameterEstimation

using Distributions: delta
using FileIO, Optim, Random,
        Statistics, Distributions, LinearAlgebra,
        Optim, Printf, LaTeXStrings, CairoMakie

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.ModelsAndData
using ..OceanTurbulenceParameterEstimation.CATKEVerticalDiffusivityModel
using ..OceanTurbulenceParameterEstimation.LossFunctions

using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

import ..OceanTurbulenceParameterEstimation.ModelsAndData: set!
import ..OceanTurbulenceParameterEstimation.LossFunctions: model_time_series

using Oceananigans.Fields: interior
using CairoMakie: Figure

export
       Parameters,
       CalibrationExperiment,
       validation_loss_reduction,
       relative_weight_options,
       set!,
       model_time_series,

       # catke_vertical_diffusivity_model_setup.jl
       InverseProblem,

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
       gradient_descent

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

struct InverseProblem{DB, PM, RW, LF, FP}
        data_batch::DB
        model::PM
        relative_weights::RW # field weights
        loss::LF
        default_parameters::FP
end

(ip::InverseProblem)(θ=ip.default_parameters) = ip.loss(ip.model, ip.data_batch, θ)

function validation_loss_reduction(calibration::InverseProblem, validation::InverseProblem, parameters::FreeParameters)
    validation_loss = validation.loss(parameters)
    calibration_loss = calibration.loss(parameters)

    default_validation_loss = validation.loss(ce.default_parameters)
    default_calibration_loss = calibration.loss(ce.default_parameters)

    validation_loss_reduction = validation_loss/default_validation_loss
    println("Parameters: $([parameters...])")
    println("Validation loss reduction: $(validation_loss_reduction)")
    println("Training loss reduction: $(calibration_loss/default_calibration_loss)")

    return validation_loss_reduction
end

include("utils.jl")
include("catke_vertical_diffusivity_model_setup.jl")
include("EKI/EKI.jl")

model_time_series(ip::InverseProblem, parameters) = model_time_series(ip.loss.ParametersToOptimize(parameters), ip.model, ip.data_batch, ip.loss.Δt)

end # module
