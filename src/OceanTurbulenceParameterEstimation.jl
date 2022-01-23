module OceanTurbulenceParameterEstimation

export
    SyntheticObservations,
    InverseProblem,
    FreeParameters,
    IdentityNormalization,
    ZScore,
    forward_map,
    forward_run!,
    observation_map,
    observation_times,
    observation_map_variance_across_time,
    ensemble_column_model_simulation,
    ConcatenatedOutputMap,
    eki,
    lognormal_with_mean_std,
    iterate!,
    EnsembleKalmanInversion,
    UnscentedKalmanInversion,
    UnscentedKalmanInversionPostprocess,
    ConstrainedNormal

include("Observations.jl")
include("EnsembleSimulations.jl")
include("TurbulenceClosureParameters.jl")
include("InverseProblems.jl")
include("EnsembleKalmanInversions.jl")

using .Observations:
    SyntheticObservations,
    ZScore,
    observation_times

using .EnsembleSimulations: ensemble_column_model_simulation

using .TurbulenceClosureParameters: FreeParameters

using .InverseProblems:
    InverseProblem,
    forward_map,
    forward_run!,
    observation_map,
    observation_map_variance_across_time,
    ConcatenatedOutputMap

using .EnsembleKalmanInversions:
    iterate!,
    EnsembleKalmanInversion,
    UnscentedKalmanInversion,
    UnscentedKalmanInversionPostprocess,
    ConstrainedNormal,
    lognormal_with_mean_std,
    NaNResampler,
    FullEnsembleDistribution,
    SuccessfulEnsembleDistribution

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

    two_day_suite_4m_paths = [lesbrary_url * "two_day_suite/8m_8m_4m_resolution/$case" * "_instantaneous_statistics.jld2"
                              for case in cases]

    two_day_suite_2m_paths = [lesbrary_url * "two_day_suite/4m_4m_2m_resolution/$case" * "_instantaneous_statistics.jld2"
                              for case in cases]
        
    dep = DataDep("two_day_suite_4m", "Ocean surface boundary layer LES data", two_day_suite_4m_paths)
    DataDeps.register(dep)

    dep = DataDep("two_day_suite_2m", "Ocean surface boundary layer LES data", two_day_suite_2m_paths)
    DataDeps.register(dep)
end

end # module

