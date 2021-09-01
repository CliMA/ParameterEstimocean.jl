module EKI

using ..ParameterEstimation
using OceanTurbulenceParameterEstimation.LossFunctions: mean_std

export 
    eki_unidimensional,
    eki_multidimensional

include("eki_unidimensional.jl")

end #module