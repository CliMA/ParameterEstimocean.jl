## Optimizing TKE parameters

using Statistics, Distributions, PyPlot
using OceanTurb, ParameterizedModelOptimizationProject, Dao
using ParameterizedModelOptimizationProject.TKEMassFluxOptimization
using ParameterizedModelOptimizationProject.TKEMassFluxOptimization: ParameterizedModel

using OceanTurb#glw/convective-adjustment-defaults

# datapath = "/Users/adelinehillier/.julia/dev/Data/three_layer_constant_fluxes_linear_hr48_Qu0.0e+00_Qb1.2e-07_f1.0e-04_Nh256_Nz128_free_convection/instantaneous_statistics.jld2"
datapath = "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu0.0e+00_Qb8.0e-08_f1.0e-04_Nh256_Nz128_free_convection/instantaneous_statistics.jld2"

# ParameterizedModel and data
tdata = TruthData(datapath)
model = ParameterizedModel(tdata, 1minute, N=32,
                    convective_adjustment = TKEMassFlux.FluxProportionalConvectiveAdjustment(),
                    eddy_diffusivities = TKEMassFlux.IndependentDiffusivities()
                    )

@free_parameters(ConvectiveAdjustmentIndependentDiffusivitesTKEParameters,
                 Cᴷc, Cᴷe,
                 Cᴰ, Cᴸᵇ, CʷwΔ, Cᴬ)

ParametersToOptimize = ConvectiveAdjustmentIndependentDiffusivitesTKEParameters
# Parameters
get_free_parameters(model)
# @free_parameters ParametersToOptimize Cᴷu Cᴷe Cᴰ Cʷu★ Cᴸᵇ
default_parameters = DefaultFreeParameters(model, ParametersToOptimize)

# Set bounds on free parameters
bounds = ParametersToOptimize(((0.01, 2.0) for p in default_parameters)...)
# bounds.Cᴷu  = (0.01, 0.5)
bounds.Cᴷc  = (0.005, 0.5)
bounds.Cᴷe  = (0.005, 0.5)
bounds.Cᴰ   = (0.01, 5.0)
bounds.Cᴸᵇ  = (0.01, 5.0)
bounds.CʷwΔ = (0.01, 10.0)
bounds.Cᴬ   = (0.01, 40.0)

# bounds.Cʷu★ = (0.01, 10.0)

targets = 1:length(tdata)
# Create loss function and negative-log-likelihood object
loss_function = LossFunction(model, tdata,
                            fields=(:T,),
                            targets=targets,
                            weights=[1.0,],
                            time_series = TimeSeriesAnalysis(tdata.t[targets], TimeAverage()),
                            profile = ValueProfileAnalysis(model.grid)
                            )

loss = LossContainer(model, tdata, loss_function)
loss(default_parameters)

# Run forward map and then compute loss from forward map output
ℱ = model_time_series(default_parameters, model, tdata, loss_function)
myloss(ℱ) = loss_function(ℱ, tdata)
myloss(ℱ)

@info "Optimizing TKE parameters..."

include("line_search_gradient_descent.jl")


f(param_vec) = loss(ParametersToOptimize(param_vec))
r = Optim.optimize(f, default_parameters)
params = Optim.minimizer(r)

# First construct global search
# Create Prior
priors = [Uniform(b...) for b in bounds]
# Determine number of function calls
functioncalls = 1000
# Define Method
method = RandomPlugin(priors, functioncalls)
# Optimize
minparam = optimize(loss, method)



println(propertynames(minparam))
# Next do gradient descent
# construct numerical gradient
∇loss(params) = gradient(loss, params)
# optimize choosing minimum from the global search for refinement
best_params = minparam
# method  = RandomLineSearch(linebounds = (0, 1e-0/norm(∇loss(best_params))), linesearches = 20)
method  = RandomLineSearch(linebounds = (0, 100.0), linesearches = 100)
bestparam = optimize(loss, ∇loss, best_params, method)
print([p for p in bestparam])
params = best_params
# params = default_parameters
p = visualize_realizations(model, tdata, 1:100:577, params)
PyPlot.savefig("default_params.png")

loss(default_parameters)
loss(best_params)

#default
# 3.35e-5



loss(minparam)
# random plugin [0.06335642383836786, 0.36492090300315955, 0.22140961407418097, 0.8116837867125876, 5.238824748510197, 7.625824258786485]
# 1.78e-6

# after GD [0.06201578909493144, 0.3723002995016638, 0.21109093788973932, 0.8117660796951753, 5.238824748510197, 7.6257996516181175]
# 1.67e-6


minparam

output_gif_directory="GP/subsample_$(subsample_frequency)/reconstruct_$(reconstruct_fluxes)/enforce_surface_fluxes_$(enforce_surface_fluxes)/train_test_same_$(train_test_same)/test_$(test_file)"
directory = pwd() * "/$(output_gif_directory)/"
mkpath(directory)
file = directory*"_output.txt"
touch(file)
o = open(file, "w")

write(o, "= = = = = = = = = = = = = = = = = = = = = = = = \n")
write(o, "Test file: $(test_file) \n")
write(o, "Output will be written to: $(output_gif_directory) \n")

close(o)
