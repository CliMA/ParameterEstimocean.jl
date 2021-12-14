using Documenter, Distributions, Test

include("test_turbulence_closure_parameters.jl")

@testset "Doctests" begin
    doctest(OceanTurbulenceParameterEstimation)
end
