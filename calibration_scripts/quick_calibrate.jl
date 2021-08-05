"""
Run this script to learn how to use OceanTurbulenceParameterEstimation.ParameterEstimation functions to calibrate
the free parameters of the TKEBasedVerticalDiffusivity closure from Oceananigans.jl to the Large Eddy simulation
ground truth generated via LESbrary.jl.
"""

#=
using ArgParse
s = ArgParseSettings()
@add_arg_table! s begin
    "--relative_weight_option"
        default = "all_but_e"
        arg_type = String
    "--free_parameters"
        default = "TKEParametersRiDependent"
        arg_type = String
end
args = parse_args(s)
relative_weight_option = args["relative_weight_option"]
free_parameter_type = args["free_parameters"]
=#

using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.TKEMassFluxModel
using OceanTurbulenceParameterEstimation.ParameterEstimation
using Statistics
using Plots
using Dao
using PyPlot
using Dates

# directory = Dates.format(Dates.now(), "yyyy-mm-dd_HH:MM:SS")

relative_weight_option = "all_but_e"
free_parameter_option = "TKEParametersRiDependent"

p = Parameters(RelevantParameters = free_parameter_options[free_parameter_option],
               ParametersToOptimize = free_parameter_options[free_parameter_option])

calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options["all_but_e"],
                                        grid_type=ZGrid,
                                        grid_size=16,
                                        Δt=10.0);

validation = dataset(merge(TwoDaySuite, SixDaySuite), p;
                                        relative_weights = relative_weight_options["all_but_e"],
                                        grid_type=ZGrid,
                                        grid_size=16,
                                        Δt=10.0);

ce = CalibrationExperiment(calibration, validation, p);

directory = "EKI/$(free_parameter_option)_$(relative_weight_option)/"
isdir(directory) || mkpath(directory)

nll = ce.calibration.nll_wrapper
nll_validation = ce.validation.nll_wrapper
initial_parameters = ce.default_parameters
parameternames = propertynames(initial_parameters)

@time nll([initial_parameters...])
# @time calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options["all_but_e"],
#                                         grid_type=ZGrid,
#                                         grid_size=64,
#                                         Δt=60.0); # 12.140647 seconds (31.52 M allocations: 2.978 GiB, 4.28% gc time)

# @time validation.nll_wrapper([initial_parameters...]) # 88.855043 seconds (184.61 M allocations: 85.167 GiB, 5.36% gc time, 35.72% compilation time) 


# using Profile
# using PProf
# @profile nll([initial_parameters...])
# pprof()
@time nll([initial_parameters...]);

using StatProfilerHTML
@profilehtml nll([initial_parameters...]);

include("calibration_scripts/visualize.jl")
animate_LESbrary_suite(ce, "try_ocean/")

#=
## Small search
set_prior_means_to_initial_parameters = true
stds_within_bounds = 2.5

# println("Initial validation loss: $(ce.validation.nll(initial_parameters))")
# validation_loss_reduction(ce, initial_parameters)


#################
## Nelder-Mead ##
#################

# Leads to negative values
@info "Running Nelder-Mead from Optim.jl..."
parameters = nelder_mead(nll, initial_parameters)
println(parameters)
validation_loss_reduction(ce, calibration.parameters.ParametersToOptimize(parameters))

##########
## BFGS ##
##########

@info "Running BFGS from Optim.jl..."
bfgs(nll, initial_parameters)
println(parameters)
validation_loss_reduction(ce, calibration.parameters.ParametersToOptimize(parameters))

#########################
## Simulated Annealing ##
#########################

@info "Running Iterative Simulated Annealing..."
prob = simulated_annealing(ce.calibration.nll, initial_parameters, ce.parameters.ParametersToOptimize; samples = 1000, iterations = 30,
                                initial_scale = 1e1,
                                final_scale = 1e-2,
                                set_prior_means_to_initial_parameters = set_prior_means_to_initial_parameters,
                                stds_within_bounds = stds_within_bounds)
params = Dao.optimal(prob.markov_chains[end]).param
validation_loss_reduction(ce, parameters)

#########################
## Other functionality ##
#########################

# Run forward map and then compute loss from forward map output
ℱ = model_time_series(default_parameters, model, tdata, loss_function)
myloss(ℱ) = loss_function(ℱ, tdata)
myloss(ℱ)
initial_parameters = ce.parameters.ParametersToOptimize(initial_parameters)

#########
## EKI ##
#########

include("calibration_scripts/EKI_setup.jl")
directory = "EKI/$(free_parameter_type)_$(relative_weight_option)/"
isdir(directory) || mkpath(directory)

plot_stds_within_bounds(nll, nll_validation, initial_parameters, directory, xrange=-3:0.25:5)
v = plot_prior_variance(nll, nll_validation, initial_parameters, directory; xrange=0.1:0.05:1.0)
plot_num_ensemble_members(nll, nll_validation, initial_parameters, directory; xrange=1:5:30)
nl = plot_observation_noise_level(nll, nll_validation, initial_parameters, directory; xrange=-2.0:0.1:3.0)

v = 4
# v = 0.8
nl = 10^(-5.0)
# v, nl = plot_prior_variance_and_obs_noise_level(nll, nll_validation, initial_parameters, directory)
# params, losses, mean_vars = eki(nll, initial_parameters; N_ens=30, N_iter=20, noise_level=nl, stds_within_bounds=v)
params = eki_better(ce.calibration.nll, ce.parameters.ParametersToOptimize, initial_parameters; N_ens=30, N_iter=20, noise_level=nl, stds_within_bounds=v, uninformed=false)
visualize_and_save(ce, ce.parameters.ParametersToOptimize(params), directory*"Visualize")
visualize_and_save(ce, ce.parameters.ParametersToOptimize(initial_parameters), directory*"InitialParameters")
run_multidimensional(kwargs) = eki_better(ce.calibration.nll, ce.parameters.ParametersToOptimize, initial_parameters; N_ens=25, N_iter=20, kwargs...)
params = run_multidimensional((noise_level=10.0^(-1.90), stds_within_bounds=4.0, uninformed=false))

run_one_dimensional(kwargs) = eki(ce.calibration.nll_wrapper, initial_parameters; N_ens=30, N_iter=20, kwargs...)
run_multidimensional(kwargs) = eki_better(ce.calibration.nll, ce.parameters.ParametersToOptimize, initial_parameters; N_ens=25, N_iter=10, kwargs...)

noises = 10 .^ collect(-2.5:0.05:-1.8)
l_cals = []
l_vals = []
for noise in noises
    params = run_one_dimensional((noise_level=noise, stds_within_bounds=0.7, uninformed=true))
    l_cal = ce.calibration.nll_wrapper(params)
    l_val = ce.validation.nll_wrapper(params)/2
    push!(l_cals, l_cal)
    push!(l_vals, l_val)
end
println("1 Noise Level")
println("CALIBRATE argmin: loss: $(l_cals[argmin(l_cals)])")
println("VALIDATE argmin: $((log10(noises[argmin(l_vals)]))) loss: $(l_vals[argmin(l_vals)])")

noises = collect(0.5:0.05:1.0)
l_cals = []
l_vals = []
for noise in noises
    params = run_one_dimensional((noise_level=1e-2, stds_within_bounds=noise, uninformed=true))
    l_cal = ce.calibration.nll_wrapper(params)
    l_val = ce.validation.nll_wrapper(params)/2
    push!(l_cals, l_cal)
    push!(l_vals, l_val)
end
println("2 Prior variance Uninformed")
println("CALIBRATE argmin: loss: $(l_cals[argmin(l_cals)])")
println("VALIDATE argmin: $(noises[argmin(l_vals)]) loss: $(l_vals[argmin(l_vals)])")

noises = 10 .^ collect(-2.5:0.05:-1.8)
l_cals = []
l_vals = []
for noise in noises
    params = run_one_dimensional((noise_level=noise, stds_within_bounds=4.0, uninformed=false))
    l_cal = ce.calibration.nll_wrapper(params)
    l_val = ce.validation.nll_wrapper(params)/2
    push!(l_cals, l_cal)
    push!(l_vals, l_val)
end
println("3 Noise Level Informed")
println("CALIBRATE argmin: loss: $(l_cals[argmin(l_cals)])")
println("VALIDATE argmin: $(log10(noises[argmin(l_vals)])) loss: $(l_vals[argmin(l_vals)])")

noises = collect(3.0:0.25:5.0)
l_cals = []
l_vals = []
for noise in noises
    params = run_one_dimensional((noise_level=1e-2, stds_within_bounds=noise, uninformed=false))
    l_cal = ce.calibration.nll_wrapper(params)
    l_val = ce.validation.nll_wrapper(params)/2
    push!(l_cals, l_cal)
    push!(l_vals, l_val)
end
println("4 Prior variance Informed")
println("CALIBRATE argmin: loss: $(l_cals[argmin(l_cals)])")
println("VALIDATE argmin: $(noises[argmin(l_vals)]) loss: $(l_vals[argmin(l_vals)])")

println(params)
println(ce.calibration.nll(initial_parameters))
println(ce.calibration.nll_wrapper(params))
println(ce.validation.nll(initial_parameters))
println(ce.validation.nll_wrapper(params))

println(initial_parameters)
println(params)

parameters = ce.default_parameters
parameters = ce.parameters.ParametersToOptimize(params)
Truth = Dict()
CATKE = Dict()
for mynll in ce.validation.nll.batch
    Truth[mynll.data.name] = mynll.data
    CATKE[mynll.data.name] = model_time_series(parameters, mynll)
end

include("calibration_scripts/visualize.jl")

whole_suite_animation(840)
anim = @animate for t=1:2:840
    p = whole_suite_animation(t)
end
Plots.gif(anim, "./$(free_parameter_type)_64_calibrated_YES.gif", fps=400)

# Parameter convergence
p = Plots.plot(title="Parameter Convergence", xlabel="Iteration", ylabel="Ensemble Covariance")
for pindex = 1:length(mean_vars[1])
    # plot!(1:length(mean_vars), [x[pindex] for x in mean_vars], lw=4, legend=:topright)
    plot!(1:length(mean_vars), [x[pindex] for x in mean_vars], label=parameter_latex_guide[parameternames[pindex]], lw=4, legend=:topright)
end
Plots.savefig(p, directory*"covariance.pdf")

# Losses
p = Plots.plot(title="Loss on Ensemble Mean", xlabel="Iteration", ylabel="Loss")
plot!(1:length(losses), losses, lw=4, legend=false)
Plots.savefig(p, directory*"loss.pdf")

##

# dependent_allbute
function convert_to_nums(A)
    B = zeros(length(A))
    for i in 1:length(A)
        if A[i] == "Inf"
            B[i] = Inf
        else
            B[i] = min(1.0, parse.(Float64, A[i]))
        end
    end
    return B
end
#
function plot_heatmap(file, vrange=0.40:0.025:0.90, nlrange=-2.5:0.1:0.5)
    abc = open(file,"r")
    line_by_line = readlines(abc)
    # A = line_by_line[2:2:end]
    A = line_by_line[2:3:end]
    B = convert_to_nums(A)

    Γθs = collect(vrange)
    Γys = 10 .^ collect(nlrange)
    losses = zeros((length(Γθs), length(Γys)))
    counter = 1
    countermax = length(Γθs)*length(Γys)
    for j in 1:length(Γys), i in 1:length(Γθs)
            losses[i, j] = B[counter]
            counter += 1
    end
    p = Plots.heatmap(Γθs, Γys, losses, xlabel=L"\Gamma_\theta", cscale=:log10, ylabel=L"\Gamma_y", size=(250,250), yscale=:log10)
    Plots.savefig(p, directory*"GammaHeatmap.pdf")
    v = Γθs[argmin(losses)[1]]
    nl = Γys[argmin(losses)[2]]
    println("loss-minimizing Γθ: $(v)")
    println("loss-minimizing log10(Γy): $(log10(nl))")
    println("$(minimum(B))")
    return v, nl
end

file = "/Users/adelinehillier/Desktop/CAdependent_allbute"
# file = "/Users/adelinehillier/Desktop/dependent_allbute"
# plot_heatmap(file)



function minslosses(file)
    abc = open(file,"r")
    line_by_line = readlines(abc)
    # A = line_by_line[2:2:end]
    A = line_by_line[1:2:end]

    B = []
    for x in A
        if !occursin(x, "NaN")
            splitx = split(x)
            splitx[1] = splitx[1][2:end]
            push!(B, [parse(Float64, y[1:end-1]) for y in splitx])
        end
    end

    # losses = [ce.calibration.nll_wrapper(params) for params in B]
    losses = []
    counter=0
    for params in B
        counter += 1
        println("$(counter)/$(length(B))")
        push!(losses, ce.calibration.nll_wrapper(params))
    end

    println(losses)
    println(minimum(losses))
    println(B[argmin(losses)])
    return B[argmin(losses)]
end

file = "/Users/adelinehillier/Desktop/CAdependent_allbute"
# file = "/Users/adelinehillier/Desktop/dependent_allbute"
minloss = minslosses(file)


##########
## Plot ##
##########

initial_parameters = ce.parameters.ParametersToOptimize(initial_parameters)

parameters = ce.default_parameters

directory="./$(free_parameter_option)_64_calibrated.gif"
animate_LESbrary_suite(ce, directory; parameters=parameters)
visualize_and_save(ce, directory; parameters=parameters)

=#
