module OceanTurbulenceParameterEstimation

export OneDimensionalTimeSeries, InverseProblem, FreeParameters

include("Observations.jl")
include("TurbulenceClosureParameters.jl")
include("InverseProblems.jl")
include("normalization.jl")

using .Observations: OneDimensionalTimeSeries
using .TurbulenceClosureParameters: FreeParameters
using .InverseProblems: InverseProblem

end # module
