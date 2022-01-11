using Documenter, Test

include("test_synthetic_observations.jl")
include("test_forward_map.jl")
include("test_ensemble_column_models.jl")
include("test_turbulence_closure_parameters.jl")

@testset "Doctests" begin
    doctest(OceanTurbulenceParameterEstimation)
end
