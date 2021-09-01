using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.ParameterEstimation
using OceanTurbulenceParameterEstimation.ModelsAndData

using OceanTurbulenceParameterEstimation.LossFunctions: t_interval, allsame

using OceanTurbulenceParameterEstimation.ModelsAndData: FourDaySuite
using OceanTurbulenceParameterEstimation.CATKEVerticalDiffusivityModel: TKEParametersRiDependent

p = Parameters(RelevantParameters = TKEParametersRiDependent,
               ParametersToOptimize = TKEParametersRiDependent)

calibration = DataSet(FourDaySuite, p; relative_weights = relative_weight_options["all_but_e"],
                                       ensemble_size = 10,
                                       Nz = 64,
                                       Δt = 10.0);

# validation = DataSet(merge(TwoDaySuite, SixDaySuite), p;
#                                         relative_weights = relative_weight_options["all_but_e"],
#                                         ensemble_size = 10,
#                                         Nz = 64,
#                                         Δt = 10.0);

# Loss on default parameters
# l0 = calibration()

# Example parameters
θ = calibration.default_parameters

# Loss on parameters θ.
# θ can be 
#   1. a vector
#   2. a FreeParameters object
#   3. a vector of parameter vectors (one for each ensemble member)
#   4. or a vector of FreeParameter objects (one for each ensemble member)
# If (1) or (2), the ensemble members are redundant and the loss is computed for just the one parameter set.

# lθ = calibration(θ)

output = model_time_series(calibration, θ)

visualize_realizations(calibration, θ)

visualize_and_save(calibration, validation, default_parameters, pwd())

eki_unidimensional(loss::DataSet, initial_parameters;
                                    noise_level = 10^(-2.0),
                                    N_ens = 10,
                                    N_iter = 15,
                                    stds_within_bounds = 0.6,
                                    informed_priors = false,
                                    objective_scale_info = false
                                    )

plot_prior_variance_and_obs_noise_level(calibration, validation, initial_parameters, directory; vrange=0.40:0.025:0.90, nlrange=-2.5:0.1:0.5)
