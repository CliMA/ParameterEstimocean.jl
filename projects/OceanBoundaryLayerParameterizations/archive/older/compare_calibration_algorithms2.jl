## Optimizing TKE parameters
using TKECalibration2021
using Plots, PyPlot
using OceanTurbulenceParameterEstimation: visualize_realizations
using Dao

LESdata = FourDaySuite # Calibration set
LESdata_validation = GeneralStrat # Validation set
RelevantParameters = TKEParametersConvectiveAdjustmentRiIndependent
ParametersToOptimize = TKEParametersConvectiveAdjustmentRiIndependent

LESdata = merge(FourDaySuite, GeneralStrat)
# dname = "calibrate_FourDaySuite_validate_GeneralStrat"
dname = "calibrate_FourDaySuiteGeneralStrat"

##
loss, initial_parameters = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize)
loss_validation, _ = custom_tke_calibration(LESdata_validation, RelevantParameters, ParametersToOptimize)

initial_parameters = ParametersToOptimize([2.1638101987647502, 0.2172594537369187, 0.4522886369267623, 0.7534625713891345, 0.4477179760916435,
        6.777679962252731, 1.2403584780163417, 1.9967245163343093])

directory = pwd() * "/TKECalibration2021Results/compare_calibration_algorithms/$(dname)/$(RelevantParameters)/"

o = open_output_file(directory)

params_dict = Dict()
loss_dict = Dict()
function writeout2(o, name, params, loss, loss_validation)
        param_vect = [params...]
        write(o, "----------- \n")
        write(o, "$(name) \n")
        write(o, "Parameters: $(param_vect) \n")
        write(o, "Loss on training: $(loss) \n")
        write(o, "Loss on validation: $(loss_validation) \n")
        params_dict[name] = param_vect
        loss_dict[name] = loss
end
writeout3(o, name, params) = writeout2(o, name, params, loss(params), loss_validation(params))

@info "Output statistics will be written to: $(directory)"

writeout3(o, "Default", initial_parameters)

# @info "Running Random Plugin..."
# random_plugin_params = random_plugin(loss, initial_parameters, ParametersToOptimize; function_calls=10000)
# writeout3(o, "Random_Plugin", random_plugin_params)
# random_plugin_params
#
# @info "Running Gradient Descent..."
# parameters = gradient_descent(loss, random_plugin_params, ParametersToOptimize; linebounds = (0, 100.0), linesearches = 10)
# writeout3(o, "Gradient_Descent", parameters)
# parameters

@info "Running Nelder-Mead from Optim.jl..."
parameters = nelder_mead(loss, initial_parameters, ParametersToOptimize)
writeout3(o, "Nelder_Mead", parameters)
parameters

@info "Running L-BFGS from Optim.jl..."
parameters = l_bfgs(loss, initial_parameters, ParametersToOptimize)
writeout3(o, "L_BFGS", parameters)

@info "Running Iterative Simulated Annealing..."
prob = simulated_annealing(loss, initial_parameters; samples=1000, iterations=10)
parameters = Dao.optimal(prob.markov_chains[end]).param
writeout3(o, "Annealing", parameters)

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

        p = visualize_realizations(case_loss.model, case_loss.data, 1:180:length(case_loss.data), best_parameters)
        # PyPlot.savefig(directory*"Test/$(case_loss.data.name).png")
        PyPlot.savefig(directory*"Train/$(case_loss.data.name).png")
end

write(o, "Loss on $(LESdata_validation) Validation Simulations: $(loss_validation(best_parameters))\n")
write(o, "Losses on Validation Simulations: \n")
for LEScase in values(LESdata_validation)
        case_loss, _ = custom_tke_calibration(LEScase, RelevantParameters, ParametersToOptimize)
        write(o, "$(case_loss.data.name): $(case_loss(best_parameters)) \n")

        p = visualize_realizations(case_loss.model, case_loss.data, 1:180:length(case_loss.data), best_parameters)
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
