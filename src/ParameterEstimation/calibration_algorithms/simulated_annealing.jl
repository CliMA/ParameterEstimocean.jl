function simulated_annealing(loss, initial_parameters, ParametersToOptimize;
                                                samples = 100,
                                             iterations = 5,
                  set_prior_means_to_initial_parameters = true,
                                     stds_within_bounds = 5,
                    annealing_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+1,
                                                                        final_scale = 1e-2,
                                                                   convergence_rate = 1.0,
                                                                    rate_adaptivity = 1.5),

                   covariance_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+1,
                                                                        final_scale = 1e+0,
                                                                   convergence_rate = 1.0,
                                                                    rate_adaptivity = 1.0),
                   unused_kwargs...
                   )

    bounds, variance = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
    prior_means = set_prior_means_to_initial_parameters ? initial_parameters : ParametersToOptimize([mean.(bounds)...])

    # Iterative simulated annealing...
    prob = anneal(loss, prior_means, variance, BoundedNormalPerturbation, bounds;
                           iterations = iterations,
                              samples = samples,
                   annealing_schedule = annealing_schedule,
                  covariance_schedule = covariance_schedule
                 )

    return prob
end
