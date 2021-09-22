module OceanTurbulenceParameterEstimation

export OneDimensionalTimeSeries, InverseProblem, FreeParameters, 
        IdentityNormalization, ZScore, forward_map

include("Observations.jl")
include("TurbulenceClosureParameters.jl")
include("InverseProblems.jl")

using .Observations: OneDimensionalTimeSeries, ZScore
using .TurbulenceClosureParameters: FreeParameters
using .InverseProblems: InverseProblem, forward_map

end # module
