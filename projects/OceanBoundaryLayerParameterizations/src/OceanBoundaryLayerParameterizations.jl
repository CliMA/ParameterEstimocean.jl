module OceanBoundaryLayerParameterizations

pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using OceanTurbulenceParameterEstimation
using Plots, LaTeXStrings, OrderedCollections

export
    # LESbrary_paths.jl
    TwoDaySuite, FourDaySuite, SixDaySuite, GeneralStrat,

    plot_stds_within_bounds, plot_prior_variance, plot_num_ensemble_members, 
    plot_observation_noise_level, plot_prior_variance_and_obs_noise_level,

    # visualization.jl
    visualize_realizations,
    visualize_and_save

include("lesbrary_paths.jl")
include("EKI_hyperparameter_search.jl")
include("visualize_profile_predictions.jl")

end # module
