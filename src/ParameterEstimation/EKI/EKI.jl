module EKI

using ..ParameterEstimation
using OceanTurbulenceParameterEstimation.LossFunctions: mean_std

export eki

include("run.jl")

end #module