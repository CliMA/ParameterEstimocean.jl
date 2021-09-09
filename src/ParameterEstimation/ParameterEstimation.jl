module ParameterEstimation

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.Observations
using ..OceanTurbulenceParameterEstimation.Models
using ..OceanTurbulenceParameterEstimation.LossFunctions

using Oceananigans
using CairoMakie: Figure

import ..OceanTurbulenceParameterEstimation.Models: set!
import ..OceanTurbulenceParameterEstimation.LossFunctions: model_time_series

export
       Parameters,
       InverseProblem,
       validation_loss_reduction,
       model_time_series,
       set!,

       # EKI
       eki

Base.@kwdef struct Parameters{T <: UnionAll}
    RelevantParameters::T
    ParametersToOptimize::T
end

struct InverseProblem{DB, PM, RW, LF, FP, PT, DT}
    observations::DB
    simulation::PM
    relative_weights::RW # field weights
    loss::LF
    default_parameters::FP
    parameters::PT
end

(ip::InverseProblem)(θ::FreeParameters = ip.default_parameters) = ip.loss(ip.simulation, ip.observations, [θ for i = 1:ensemble_size(m)])
(ip::InverseProblem)(θ::Vector{<:Number} = ip.default_parameters) = ip.loss(ip.simulation, ip.observations, ip.parameters.ParametersToOptimize(θ))
(ip::InverseProblem)(θ::Vector{<:Vector} = ip.default_parameters) = ip.loss(ip.simulation, ip.observations, ip.parameters.ParametersToOptimize.(θ))
(ip::InverseProblem)(θ::Matrix = ip.default_parameters) = ip.loss(ip.simulation, ip.observations, [θ[:,i] for i in 1:size(θ, 2)])

model_time_series(ip::InverseProblem, parameters) = model_time_series(ip.simulation, ip.observations, ip.parameters.ParametersToOptimize(parameters))

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

include("inverse_problem.jl")
include("EKI/EKI.jl")

using .EKI

end # module
