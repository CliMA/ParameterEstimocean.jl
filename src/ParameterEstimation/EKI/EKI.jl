module EKI

using ..ParameterEstimation
using OceanTurbulenceParameterEstimation.LossFunctions: mean_std
using LaTeXStrings

export 
    eki_unidimensional,
    eki_multidimensional,

    plot_stds_within_bounds,
    plot_prior_variance,
    plot_num_ensemble_members,
    plot_observation_noise_level,
    plot_prior_variance_and_obs_noise_level


include("eki_unidimensional.jl")

end #module