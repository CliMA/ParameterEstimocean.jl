using Test
using DataDeps
using OceanTurbulenceParameterEstimation

always_accept = get(ENV, "DATADEPS_ALWAYS_ACCEPT", false)
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

function run_datadep_test(dep_str)
    @testset "DataDep $dep_str" begin
        @datadep_str dep_str
        @test true
    end
    return nothing
end

for dep in ["two_day_suite_2m", 
            "two_day_suite_4m",
            "four_day_suite_2m",
            "four_day_suite_4m",
            "six_day_suite_2m",
            "six_day_suite_4m"]

    run_datadep_test(dep)
end

ENV["DATADEPS_ALWAYS_ACCEPT"] = always_accept

