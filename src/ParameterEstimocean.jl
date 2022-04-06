module ParameterEstimocean

export
    SyntheticObservations,
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
    eki,
    lognormal,
    ScaledLogitNormal,
    iterate!,
    EnsembleKalmanInversion,
    Resampler,
    FullEnsembleDistribution,
    SuccessfulEnsembleDistribution,
    ConstrainedNormal

include("Utils.jl")
include("Transformations.jl")
include("Observations.jl")
include("Parameters.jl")
include("EnsembleSimulations.jl")
include("InverseProblems.jl")
include("EnsembleKalmanInversions.jl")

using .Utils
using .Transformations
using .Observations
using .EnsembleSimulations
using .Parameters
using .InverseProblems
using .EnsembleKalmanInversions

#####
##### Data!
#####

using DataDeps

function __init__()
    # Register LESbrary data
    lesbrary_url = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/glw/stokesian-lesbrary/LESbrary/idealized/"

    cases = ["free_convection",
             "weak_wind_strong_cooling", 
             "med_wind_med_cooling", 
             "strong_wind_weak_cooling", 
             "strong_wind",
             "strong_wind_no_rotation"]

    two_day_suite_url = lesbrary_url * "two_day_suite/"

    glom_url(suite, resolution, case) = string(lesbrary_url,
                                               suite, "/", resolution, "_resolution/",
                                               case, "_instantaneous_statistics.jld2")

    two_day_suite_4m_paths  = [glom_url( "two_day_suite", "4m", case) for case in cases]
    
    dep = DataDep("two_day_suite_4m", "Idealized 2 day simulation data with 4m horizontal resolution", two_day_suite_4m_paths)
    DataDeps.register(dep)
end

end # module

