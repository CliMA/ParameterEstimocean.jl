## Optimizing TKE parameters
using TKECalibration2021
using Plots

@free_parameters(ConvectiveAdjustmentParameters,
                 Cᴬu, Cᴬc, Cᴬe)

include("compare_calibration_algorithms_setup.jl")

p = Parameters(RelevantParameters = TKEParametersRiDependent,
               ParametersToOptimize = TKEParametersRiDependent
              )
rw = relative_weight_options["all_e"]
rw = Dict(:T => 1.0, :U => 0.5, :V => 0.5, :e => 0.0)

calibration = dataset(FourDaySuite, p; relative_weights = rw, N=64, Δt=10.0);
validation = dataset(merge(TwoDaySuite, SixDaySuite), p; relative_weights = relative_weight_options["all_but_e"], N=64, Δt=10.0);

ce = CalibrationExperiment(calibration, validation, p)

initial_parameters = ce.default_parameters

ce.validation.loss(initial_parameters)

loss = calibration.loss

# @time ce.validation.loss([initial_parameters...])
# @time ce.calibration.loss([initial_parameters...])
# single = dataset(FourDaySuite["4d_free_convection"], p; relative_weights = relative_weight_options["all_e"]);
# @time single.loss([initial_parameters...])

validation_loss_reduction(ce, initial_parameters)
ce.validation.loss(initial_parameters)

## Large search
# set_prior_means_to_initial_parameters = false
# stds_within_bounds = 5
# dname = "calibrate_FourDaySuite_validate_TwoDaySuiteSixDaySuite/prior_mean_center_bounds/$(relative_weights_option)_weights"

## Small search
set_prior_means_to_initial_parameters = true
stds_within_bounds = 3
# dname = "calibrate_FourDaySuite_validate_TwoDaySuiteSixDaySuite/prior_mean_optimal/$(relative_weights_option)_weights"

# xs = collect(0.001:0.001:0.025)
# ll = Dict()
# for x in xs
#         initial_parameters.Cᴬc = 0.6706
#         ll[x] = loss_validation(initial_parameters)
# end
# Plots.plot(ll, yscale = :log10)
# argmin(ll)

##
# directory = pwd() * "/TKECalibration2021Results/compare_calibration_algorithms/$(dname)/$(RelevantParameters)/"
#
# o = open_output_file(directory)
#
# params_dict = Dict()
# loss_dict = Dict()
# function writeout2(o, name, params, loss, loss_validation)
#         param_vect = [params...]
#         write(o, "----------- \n")
#         write(o, "$(name) \n")
#         write(o, "Parameters: $(param_vect) \n")
#         write(o, "Loss on training: $(loss) \n")
#         write(o, "Loss on validation: $(loss_validation) \n")
#         params_dict[name] = param_vect
#         loss_dict[name] = loss_validation
# end
# writeout3(o, name, params) = writeout2(o, name, params, loss(params), loss_validation(params))
#
# @info "Output statistics will be written to: $(directory)"
#
# writeout3(o, "Default", initial_parameters)
#
# @info "Running Random Plugin..."
# random_plugin_params = random_plugin(loss, initial_parameters; function_calls=1000)
# writeout3(o, "Random_Plugin", random_plugin_params)
# println("Random Plugin", random_plugin_params)
#
# @info "Running Gradient Descent..."
# parameters = gradient_descent(loss, random_plugin_params; linebounds = (0, 100.0), linesearches = 10)
# writeout3(o, "Gradient_Descent", parameters)
# println("Gradient_Descent", parameters)
#
#
# @info "Running Nelder-Mead from Optim.jl..."
# parameters = nelder_mead(loss, initial_parameters)
# writeout3(o, "Nelder_Mead", parameters)
# println(parameters)
# println(validation_loss_reduction(ce, ce.parameters.ParametersToOptimize(parameters)))
#
# loss(initial_parameters)
# loss([parameters...])
# loss_validation(initial_parameters)
# loss_validation([parameters...])
#
# @info "Running L-BFGS from Optim.jl..."
# parameters = l_bfgs(loss, initial_parameters)
# writeout3(o, "L_BFGS", parameters)
#
# stds_within_bounds = 5
@info "Running Iterative Simulated Annealing..."
prob = simulated_annealing(loss, initial_parameters; samples = 10, iterations = 3,
                                initial_scale = 1e1,
                                final_scale = 1e-1,
                                set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                stds_within_bounds = stds_within_bounds)
parameters = Dao.optimal(prob.markov_chains[end]).param
writeout3(o, "Annealing", parameters)
println([parameters...])
println(validation_loss_reduction(ce, ce.parameters.ParametersToOptimize(parameters)))

# initial_parameters = ParametersToOptimize([0.029469308779054255, 31.6606181722508, 416.89781702903394])
# propertynames(initial_parameters)
# loss(initial_parameters)
# loss_validation(initial_parameters)
# loss(ParametersToOptimize(parameters))
# loss_validation(ParametersToOptimize(parameters))
# println([parameters...])
#
# @info "Running Ensemble Kalman Inversion..."

throw(exception)

parameters = ensemble_kalman_inversion(loss, initial_parameters; N_ens = 50, N_iter = 10,
                                set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                stds_within_bounds = 10)
# writeout3(o, "EKI", ParametersToOptimize(parameters))

validation_losses = Dict()
initial_validation_loss = loss_validation(initial_parameters)
for x = 0.0:0.1:1.0
        println(x)
        relative_weights = Dict(:T => 1.0, :U => x, :V => x, :e => x)
        loss, _ = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize;
                                        loss_closure = loss_closure,
                                        relative_weights = relative_weights)
        prob = simulated_annealing(loss, initial_parameters; samples = 100, iterations = 3,
                                        initial_scale = 1e0,
                                        set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                        stds_within_bounds = stds_within_bounds)
        parameters = Dao.optimal(prob.markov_chains[end]).param
        println(parameters)
        loss_reduction =  loss_validation(ParametersToOptimize(parameters)) / initial_validation_loss
        validation_losses[x] = loss_reduction
        println(loss_reduction)
end
p = Plots.plot(validation_losses, ylabel="Validation loss reduction (Final / Initial)", xlabel="relative weight for U, V, e (where T -> 1)", title = "Loss reduction vs. U, V, e relative weight", legend=false, lw=3)
Plots.savefig(p, "____relative_weight_UVe.pdf")



prob = simulated_annealing(loss, initial_parameters; samples = 500, iterations = 3,
                                initial_scale = 1e1,
                                set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                stds_within_bounds = stds_within_bounds)
parameters = Dao.optimal(prob.markov_chains[end]).param


propertynames(ParametersToOptimize(initial_parameters))

loss(initial_parameters)
loss_validation(initial_parameters)
println(parameters)
loss(parameters)
loss_validation(parameters)

initial_parameters = ParametersToOptimize([0.0057, 0.005, 0.005])

best_parameters = ParametersToOptimize(initial_parameters)
directory = "TKECalibration2021Results/annealing_visuals_smaller_CA/"


##
@info "Visualizing final parameter values for each calibration method..."
methodnames = [keys(params_dict)...]
parameters
# x-axis: sort calibration method names by the loss, highest to lowest (theoretically Default should come first)
sort_key(methodname) = loss_dict[methodname]
methodnames = sort([keys(loss_dict)...], by=sort_key, rev=true)

isdir(directory*"Parameters/") || mkdir(directory*"Parameters/")
for i in 1:length(parameters)
        paramname = propertynames(parameters)
        parameter_vals = [params_dict[name][i] for name in methodnames]
        p = Plots.plot(methodnames, parameter_vals, size=(600,150), linewidth=3, xrotation = 60, label=parameter_latex_guide[paramname[i]])
        Plots.savefig(directory*"Parameters/$(parameter_latex_guide[paramname[i]]).pdf")
end

@info "Visualizing final loss values for each calibration method..."

loss_vals = [loss_dict[name] for name in methodnames]
p = Plots.plot(methodnames, loss_vals, size=(600,150), linewidth=3, xrotation = 60, label="loss", yscale=:log10, color = :purple)
Plots.savefig(directory*"losses.png")

@info "Visualizing how the parameters perform on new data..."

# Find the overall best loss-minimizing parameter
best_method = argmin(loss_dict)
best_parameters = ParametersToOptimize(params_dict[best_method])

# best_method = "Nelder_Mead"
# best_parameters = ParametersToOptimize([0.494343889780388, 0.5671815040873687, 0.8034339015426114, 0.40412711476911073, 0.23935082563117294, 7.594543282811973, 0.19964793087118093, 3.01077631309058])

isdir(directory*"Test/") || mkdir(directory*"Test/")
isdir(directory*"Train/") || mkdir(directory*"Train/")
write(o, " \n")
write(o, "Best method: $(best_method)\n")
write(o, "Best loss: $(loss_dict[best_method])\n")
write(o, "Best parameters: \n")
write(o, "$(best_parameters) \n")

write(o, "Losses on Calibration Simulations: \n")
for LEScase in values(LESdata)
        case_loss, _ = custom_tke_calibration(LEScase, RelevantParameters, ParametersToOptimize)
        write(o, "$(case_loss.data.name): $(case_loss(best_parameters)) \n")

        p = visualize_predictions(case_loss.model, case_loss.data, 12:180:length(case_loss.data), best_parameters)
        PyPlot.savefig(directory*"Train/$(case_loss.data.name).png")
end

write(o, "Loss on $(LESdata_validation) Validation Simulations: $(loss_validation(best_parameters))\n")
write(o, "Losses on Validation Simulations: \n")
for LEScase in values(LESdata_validation)
        case_loss, _ = custom_tke_calibration(LEScase, RelevantParameters, ParametersToOptimize)
        write(o, "$(case_loss.data.name): $(case_loss(best_parameters)) \n")

        p = visualize_predictions(case_loss.model, case_loss.data, 12:180:length(case_loss.data), best_parameters)
        PyPlot.savefig(directory*"Test/$(case_loss.data.name).png")
end

# Close output.txt
close(o)

##
# initial_parameters = ParametersToOptimize([2.3923033609398985, 0.20312574763086733, 0.46858459323259577,
#                                            0.4460275753651033, 0.5207833999203864, 5.368922290345999,
#                                            1.1855706525110876, 2.6304133954266207])

# using above initial parameters, Nelder-Mead finds:
# initial_parameters = ParametersToOptimize([2.1638101987647502, 0.2172594537369187, 0.4522886369267623, 0.7534625713891345, 0.4477179760916435,
#         6.777679962252731, 1.2403584780163417, 1.9967245163343093])
