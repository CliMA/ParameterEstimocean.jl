## Optimizing TKE parameters

using Statistics, Distributions, PyPlot, Plots
using OceanTurb, OceanTurbulenceParameterEstimation, Dao
using OceanTurbulenceParameterEstimation.TKEMassFluxOptimization
using OceanTurbulenceParameterEstimation.TKEMassFluxOptimization: ParameterizedModel
using Optim

to_calibrate = "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu0.0e+00_Qb8.0e-08_f1.0e-04_Nh256_Nz128_free_convection/instantaneous_statistics.jld2"
to_test = ["/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu0.0e+00_Qb1.2e-07_f1.0e-04_Nh256_Nz128_free_convection/instantaneous_statistics.jld2",
        "/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu1.0e-03_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind/instantaneous_statistics.jld2",
        "/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu2.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation/instantaneous_statistics.jld2",
        "/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu3.0e-04_Qb1.0e-07_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling/instantaneous_statistics.jld2",
        "/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu8.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling/instantaneous_statistics.jld2",
        "/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu1.0e-03_Qb-4.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_heating/instantaneous_statistics.jld2"]

include("my_models.jl")
build_model = tke_free_convection_independent_diffusivities
# build_model = conv_adj_ri_dependent_diffusivites

mymodel = build_model(to_calibrate);
tdata = mymodel.tdata
model = mymodel.model
get_free_parameters(model)
ParametersToOptimize = mymodel.ParametersToOptimize
default_parameters = mymodel.default_parameters
bounds = mymodel.bounds
loss_function, loss = build_loss(model, tdata)
loss(default_parameters)

fieldnames(ParametersToOptimize)
default_parameters
set!(model, default_parameters)
model
@free_parameters testing Cᴰ Cᴷc
set!(model, testing([1.0,2.0]))
model

# Run forward map and then compute loss from forward map output
ℱ = model_time_series(default_parameters, model, tdata, loss_function)
myloss(ℱ) = loss_function(ℱ, tdata)
myloss(ℱ)

##

directory = pwd() * "/compare_calibration_methods/calibrate_$(tdata.name)_$(length(tdata.t))/$(build_model)/"
isdir(directory) || mkdir(directory)
file = directory*"output.txt"
touch(file)
o = open(file, "w")
write(o, "Calibrating $(tdata.name) scenario \n")

@info "Output statistics will be written to: $(file)"

function saveplot(params, name)
        p = visualize_realizations(model, tdata, 1:180:length(tdata), params)
        PyPlot.savefig(directory*name*".png")
end

params_dict = Dict()
loss_dict = Dict()
function writeout(o, name, params)
        param_vect = [params...]
        loss_value = loss(params)
        write(o, "----------- \n")
        write(o, "$(name) \n")
        write(o, "Parameters: $(param_vect) \n")
        write(o, "Loss: $(loss_value) \n")
        saveplot(params, name)
        params_dict[name] = param_vect
        loss_dict[name] = loss_value
end
writeout(o, "Default", default_parameters)

@info "Running Nelder-Mead from Optim.jl..."

loss_wrapper(param_vec) = loss(ParametersToOptimize(param_vec))
r = Optim.optimize(loss_wrapper, [default_parameters...])
params = ParametersToOptimize(Optim.minimizer(r))
writeout(o, "Nelder_Mead", params)

@info "Running L-BFGS from Optim.jl..."

loss_wrapper(param_vec) = loss(ParametersToOptimize(param_vec))
r = Optim.optimize(loss_wrapper, [default_parameters...], LBFGS())
params = ParametersToOptimize(Optim.minimizer(r))
writeout(o, "L_BFGS", params)

@info "Running Random Plugin..."

include("../examples/line_search_gradient_descent.jl")
priors = [Uniform(b...) for b in bounds]
functioncalls = 1000
method = RandomPlugin(priors, functioncalls)
minparam = optimize(loss, method);
writeout(o, "Random_Plugin", minparam)

@info "Running Gradient Descent..."

∇loss(params) = gradient(loss, params) # numerical gradient
method  = RandomLineSearch(linebounds = (0, 100.0), linesearches = 100)
bestparam = optimize(loss, ∇loss, minparam, method);
writeout(o, "Gradient_Descent", bestparam)

@info "Running Iterative Simulated Annealing..."

variance = Array(ParametersToOptimize((0.1 * bound[2] for bound in bounds)...))
prob = anneal(loss, default_parameters, variance, BoundedNormalPerturbation, bounds;
             iterations = 10,
                samples = 1000,
     annealing_schedule = AdaptiveExponentialSchedule(initial_scale=100.0, final_scale=1e-3, convergence_rate=1.0),
    covariance_schedule = AdaptiveExponentialSchedule(initial_scale=1.0,   final_scale=1e-3, convergence_rate=0.1),
);
params = optimal(prob.markov_chains[end]).param
writeout(o, "Annealing", params)

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
write(o, "Losses on 2-Day Suite Simulations: \n")
for test_file in to_test

        mymodel_test = build_model(test_file);
        tdata_test = mymodel_test.tdata
        model_test = mymodel_test.model
        loss = build_loss(model_test, tdata_test)

        write(o, "$(tdata.name): $(loss(params)) \n")

        p = visualize_realizations(model_test, tdata_test, 1:90:length(tdata_test), best_parameters)
        PyPlot.savefig(directory*"Test/$(tdata_test.name).png")
end


# Close output.txt
close(o)
