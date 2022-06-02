"""
Main module for `ParameterEstimocean.jl` -- a Julia software for parameter estimation
for Oceananigans models using Ensemble Kalman Inversion.
"""
module ParameterEstimocean

export
    SyntheticObservations,
    BatchedSyntheticObservations,
    InverseProblem,
    FreeParameters,
    Transformation,
    SpaceIndices,
    TimeIndices,
    RescaledZScore,
    ZScore,
    forward_map,
    forward_run!,
    observation_map,
    observation_times,
    observation_map_variance_across_time,
    ensemble_column_model_simulation,
    ConcatenatedOutputMap,
    ConcatenatedVectorNormMap,
    eki,
    lognormal,
    ScaledLogitNormal,
    iterate!,
    EnsembleKalmanInversion,
    Resampler,
    FullEnsembleDistribution,
    SuccessfulEnsembleDistribution,
    ConstantConvergence

include("Utils.jl")
include("Transformations.jl")
include("Observations.jl")
include("Parameters.jl")
include("EnsembleSimulations.jl")
include("InverseProblems.jl")
include("EnsembleKalmanInversions.jl")
include("PseudoSteppingSchemes.jl")

using .Utils
using .Transformations
using .Observations
using .EnsembleSimulations
using .Parameters
using .InverseProblems
using .EnsembleKalmanInversions
using .PseudoSteppingSchemes

#####
##### Data!
#####

using DataDeps

function __init__()
    # Register LESbrary data
    lesbrary_url = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/glw/lesbrary2/LESbrary/idealized/"

    cases = ["free_convection",
             "weak_wind_strong_cooling",
             "strong_wind_weak_cooling",
             "strong_wind",
             "strong_wind_no_rotation"]

    two_day_suite_url = lesbrary_url * "two_day_suite/"

    glom_url(suite, resolution, case) = string(lesbrary_url,
                                               suite, "/", resolution, "_resolution/",
                                               case, "_instantaneous_statistics.jld2")

    two_day_suite_1m_paths  = [glom_url( "two_day_suite", "2m_2m_1m", case) for case in cases]
    two_day_suite_2m_paths  = [glom_url( "two_day_suite", "4m_4m_2m", case) for case in cases]
    two_day_suite_4m_paths  = [glom_url( "two_day_suite", "8m_8m_4m", case) for case in cases]
    four_day_suite_1m_paths = [glom_url("four_day_suite", "2m_2m_1m", case) for case in cases]
    four_day_suite_2m_paths = [glom_url("four_day_suite", "4m_4m_2m", case) for case in cases]
    four_day_suite_4m_paths = [glom_url("four_day_suite", "8m_8m_4m", case) for case in cases]
    six_day_suite_1m_paths  = [glom_url( "six_day_suite", "2m_2m_1m", case) for case in cases]
    six_day_suite_2m_paths  = [glom_url( "six_day_suite", "4m_4m_2m", case) for case in cases]
    six_day_suite_4m_paths  = [glom_url( "six_day_suite", "8m_8m_4m", case) for case in cases]

    dep = DataDep("two_day_suite_1m", "Idealized 2 day simulation data with 1m vertical resolution", two_day_suite_1m_paths)
    DataDeps.register(dep)

    dep = DataDep("two_day_suite_2m", "Idealized 2 day simulation data with 2m vertical resolution", two_day_suite_2m_paths)
    DataDeps.register(dep)

    dep = DataDep("two_day_suite_4m", "Idealized 2 day simulation data with 4m vertical resolution", two_day_suite_4m_paths)
    DataDeps.register(dep)

    dep = DataDep("four_day_suite_1m", "Idealized 4 day simulation data with 1m vertical resolution", four_day_suite_1m_paths)
    DataDeps.register(dep)

    dep = DataDep("four_day_suite_2m", "Idealized 4 day simulation data with 2m vertical resolution", four_day_suite_2m_paths)
    DataDeps.register(dep)

    dep = DataDep("four_day_suite_4m", "Idealized 4 day simulation data with 4m vertical resolution", four_day_suite_4m_paths)
    DataDeps.register(dep)

    dep = DataDep("six_day_suite_1m", "Idealized 6 day simulation data with 1m vertical resolution", six_day_suite_1m_paths)
    DataDeps.register(dep)

    dep = DataDep("six_day_suite_2m", "Idealized 6 day simulation data with 2m vertical resolution", six_day_suite_2m_paths)
    DataDeps.register(dep)

    dep = DataDep("six_day_suite_4m", "Idealized 6 day simulation data with 4m vertical resolution", six_day_suite_4m_paths)
    DataDeps.register(dep)
end

end # module

