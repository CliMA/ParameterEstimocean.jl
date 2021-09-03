## Optimizing TKE parameters
using TKECalibration2021
using Plots, PyPlot
using OceanTurbulenceParameterEstimation: visualize_predictions

             LESdata = FourDaySuite # Calibration set
  LESdata_validation = GeneralStrat # Validation set
  RelevantParameters = TKEFreeConvectionConvectiveAdjustmentRiIndependent
ParametersToOptimize = TKEFreeConvectionConvectiveAdjustmentRiIndependent

dname = "calibrate_FourDaySuite_validate_GeneralStrat"

##
loss, initial_parameters = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize)

directory = pwd() * "/TKECalibration2021Results/compare_calibration_algorithms/$(dname)/$(RelevantParameters)/"

o = open_output_file(directory)

params_dict = Dict()
loss_dict = Dict()
function writeout2(o, name, params, loss)
        param_vect = [params...]
        write(o, "----------- \n")
        write(o, "$(name) \n")
        write(o, "Parameters: $(param_vect) \n")
        write(o, "Loss: $(loss) \n")
        params_dict[name] = param_vect
        loss_dict[name] = loss
end

@info "Output statistics will be written to: $(directory)"

writeout2(o, "Default", initial_parameters, loss(initial_parameters))

@info "Running Nelder-Mead from Optim.jl..."
params = nelder_mead(ℒ, initial_parameters, ParametersToOptimize)
writeout2(o, "Nelder_Mead", params, loss(params))

@info "Running L-BFGS from Optim.jl..."
params = l_bfgs(ℒ, initial_parameters, ParametersToOptimize)
writeout2(o, "L_BFGS", params, loss(params))

@info "Running Random Plugin..."
random_plugin_params = random_plugin(loss, initial_parameters, ParametersToOptimize; function_calls=10000)
writeout2(o, "Random_Plugin", random_plugin_params, loss(random_plugin_params))

@info "Running Gradient Descent..."
params = gradient_descent(loss, random_plugin_params, ParametersToOptimize; linebounds = (0, 100.0), linesearches = 100)
writeout2(o, "Gradient_Descent", params, loss(params))

@info "Running Iterative Simulated Annealing..."
prob = simulated_annealing(loss, initial_parameters; samples=1000, iterations=10)
params = optimal(prob.markov_chains[end]).param
writeout2(o, "Annealing", params, loss(params))

@info "Visualizing final parameter values for each calibration method..."

methodnames = [keys(params_dict)...]

# x-axis: sort calibration method names by the loss, highest to lowest (theoretically Default should come first)
sort_key(methodname) = loss_dict[methodname]
methodnames = sort([keys(loss_dict)...], by=sort_key, rev=true)

isdir(directory*"Parameters/") || mkdir(directory*"Parameters/")
for i in 1:length(params)
        paramname = propertynames(params)
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

isdir(directory*"Test/") || mkdir(directory*"Test/")
write(o, " \n")
write(o, "Best method: $(best_method)\n")
write(o, "Best loss: $(loss_dict[best_method])\n")
write(o, "Best parameters: \n")
write(o, "$(best_parameters) \n")

write(o, "Losses on $(LESdata) Simulations: \n")
for LEScase in values(LESdata)
        case_loss, _ = custom_tke_calibration(LEScase, RelevantParameters, ParametersToOptimize)
        write(o, "$(case_loss.data.name): $(case_loss(best_parameters)) \n")

        p = visualize_predictions(case_loss.model, case_loss.data, 1:180:length(case_loss.data), best_parameters)
        PyPlot.savefig(directory*"Test/$(case_loss.data.name).png")
end

loss_validation, _ = custom_tke_calibration(LESdata_validation, RelevantParameters, ParametersToOptimize)
write(o, "Loss on $(LESdata_validation) Validation Simulations: $(loss_validation(best_parameters))\n")

write(o, "Losses on $(LESdata) Simulations: \n")
for LEScase in values(LESdata_validation)
        case_loss, _ = custom_tke_calibration(LEScase, RelevantParameters, ParametersToOptimize)
        write(o, "$(case_loss.data.name): $(case_loss(best_parameters)) \n")

        p = visualize_predictions(case_loss.model, case_loss.data, 1:180:length(case_loss.data), best_parameters)
        PyPlot.savefig(directory*"Test/$(case_loss.data.name).png")
end

# Close output.txt
close(o)
