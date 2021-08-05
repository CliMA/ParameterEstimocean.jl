## Optimizing TKE parameters
using TKECalibration2021
using Dao
using Statistics
using Plots

LESdata = TwoDaySuite # Calibration set
# LESdata_validation = FourDaySuite # Validation set
RelevantParameters = TKEParametersRiIndependent
ParametersToOptimize = TKEParametersRiIndependent

relative_weights = Dict(:T => 1.0,
                    :U => 1e-2,
                    :V => 1e-2,
                    :e => 1e-4)

## Large search
set_prior_means_to_initial_parameters = false
plot_prefix = "annealing__prior_mean_center_bound"

## Small search
# set_prior_means_to_initial_parameters = true
# plot_prefix = "annealing__prior_mean_optimal"

function loss_closure(nll)
        ℒ(parameters::ParametersToOptimize) = nll(parameters)
        ℒ(parameters::Vector) = nll(ParametersToOptimize([parameters...]))
        return ℒ
end

nll, initial_parameters = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize;
                                        loss_closure = loss_closure,
                                        relative_weights = relative_weights)

initial_parameters = ParametersToOptimize([0.1320799067908237, 0.21748565946199314, 0.051363488558909924, 0.5477193236638974, 0.8559038503413254, 3.681157252463703, 2.4855193201082426])
nll(initial_parameters)

function simulated_annealing_experimental(nll, initial_parameters;
                                                samples = 500,
                                             iterations = 5,
                  set_prior_means_to_initial_parameters = true,
                                     stds_within_bounds = 5,
                                          initial_scale = 1e1,
                                            final_scale = 1e-2,
                                       convergence_rate = 1.0,
                                        rate_adaptivity = 1.5
                   )
   bounds, variance = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
   initial_parameters = set_prior_means_to_initial_parameters ? initial_parameters : [mean.(bounds)...]

    # Iterative simulated annealing...
    prob = anneal(nll, initial_parameters, variance, BoundedNormalPerturbation, bounds;
                           iterations = iterations,
                              samples = samples,
                   annealing_schedule = AdaptiveAlgebraicSchedule(   initial_scale = initial_scale,
                                                                       final_scale = final_scale,
                                                                  convergence_rate = convergence_rate,
                                                                   rate_adaptivity = rate_adaptivity),
                  covariance_schedule = AdaptiveAlgebraicSchedule(   initial_scale = 1e+1,
                                                                       final_scale = 1e+0,
                                                                  convergence_rate = 1.0,
                                                                   rate_adaptivity = 1.0)
                 );

    return prob
end

initial_parameters = ParametersToOptimize([0.1320799067908237, 0.21748565946199314, 0.051363488558909924, 0.5477193236638974, 0.8559038503413254, 3.681157252463703, 2.4855193201082426])
bounds, _ = get_bounds_and_variance(initial_parameters; stds_within_bounds = 0);
annealing_initial_parameters = ParametersToOptimize([mean.(bounds)...])
function loss_reduction(kwargs)
    println(kwargs)
    initial_loss = nll(annealing_initial_parameters)
    prob = simulated_annealing_experimental(nll, annealing_initial_parameters; samples=200, iterations=5, set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters, kwargs...);
    final_parameters = Dao.optimal(prob.markov_chains[end]).param
    println([final_parameters...])
    loss_reduction = nll(final_parameters) / initial_loss
    println(loss_reduction)
    return loss_reduction
end

## Stds within bounds

# loss_reduction((stds_within_bounds = 3, set_prior_means_to_initial_parameters=false))
loss_reductions = Dict()
for stds_within_bounds = 3:10.0
    loss_reductions[stds_within_bounds] = loss_reduction((stds_within_bounds = stds_within_bounds, ))
end
p = Plots.plot(loss_reductions, title="Loss reduction versus prior std's within bounds", xlabel="Loss reduction (Final / Initial)", ylabel="prior std's spanned by bound width", legend=false, lw=3)
Plots.savefig(p, plot_prefix*"____stds_within_bounds.pdf")

## Number of samples

loss_reductions = Dict()
for samples = 50:50:1000
    loss_reductions[samples] = loss_reduction((samples = samples,))
end
p = Plots.plot(loss_reductions, title="Loss reduction versus number of samples", xlabel="number of samples", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
Plots.savefig(p, plot_prefix*"____samples.pdf")

## Number of iterations

loss_reductions = Dict()
for iterations = 1:20
    loss_reductions[iterations] = loss_reduction((iterations = iterations,))
end
p = Plots.plot(loss_reductions, title="Loss reduction versus number of iterations", xlabel="number of iterations", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
Plots.savefig(p, plot_prefix*"____iterations.pdf")

## Initial scale

loss_reductions = Dict()
for log_initial_scale = 0.0:0.25:0.25:2.0
    loss_reductions[log_initial_scale] = loss_reduction((initial_scale = 10.0^log_initial_scale,))
end
p = Plots.plot(loss_reductions, title="Loss reduction vs. initial scale (annealing schedule)", xlabel="log₁₀(Initial scale)", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
Plots.savefig(p, plot_prefix*"____initial_scale.pdf")

## Final scale

loss_reductions = Dict()
for log_final_scale = -2.0:0.25:1.0
    loss_reductions[log_final_scale] = loss_reduction((final_scale = 10.0^log_final_scale,))
end
p = Plots.plot(loss_reductions, title="Loss reduction vs. final scale (annealing schedule)", xlabel="log₁₀(Initial scale)", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
Plots.savefig(p, plot_prefix*"____final_scale.pdf")

loss_reductions = Dict()
for convergence_rate = 0.0:0.25:3.0
    loss_reductions[convergence_rate] = loss_reduction((convergence_rate = convergence_rate,))
end
p = Plots.plot(loss_reductions, title="Loss reduction vs. convergence rate (annealing schedule)", xlabel="Convergence rate", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
Plots.savefig(p, plot_prefix*"____convergence_rate.pdf")

loss_reductions = Dict()
for rate_adaptivity = 0.0:0.25:3.0
    loss_reductions[rate_adaptivity] = loss_reduction((rate_adaptivity = rate_adaptivity,))
end
p = Plots.plot(loss_reductions, title="Loss reduction vs. rate adaptivity (annealing schedule)", xlabel="Rate_adaptivity", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
Plots.savefig(p, plot_prefix*"____rate_adaptivity.pdf")
