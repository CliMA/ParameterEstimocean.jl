module EKI

using ..ParameterEstimation
using OceanTurbulenceParameterEstimation.LossFunctions
using OceanTurbulenceParameterEstimation.LossFunctions: mean_std

using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
using Distributions, LinearAlgebra, Random, Statistics


export 
        # forward_map_output.jl
        SqrtLossForwardMapOutput,
        ConcatenatedProfilesForwardMapOutput,

        # run.jl
        eki

abstract type AbstractForwardMapOutput end

include("forward_map_output.jl")
include("run.jl")

end #module