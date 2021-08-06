using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.TKEMassFluxModel
using OceanTurbulenceParameterEstimation.ParameterEstimation

using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

using Distributions
using LinearAlgebra
using Random
using Dao
using Plots
using PyPlot
using ArgParse

#=
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

relative_weight_option = "all_but_e"
free_parameter_type = "TKEParametersRiDependent"

relative_weight = Dict(:T => 1.0, :U => 0.5, :V => 0.5, :e => 0.0)

include("calibration_scripts/EKI_setup.jl")

directory = "EKI/$(free_parameter_type)_$(relative_weight_option)/"
isdir(directory) || mkpath(directory)

p = Parameters(RelevantParameters = free_parameter_options[free_parameter_option],
               ParametersToOptimize = free_parameter_options[free_parameter_option])

calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options["all_but_e"], grid_type=ZGrid, grid_size=64, Δt=10.0);
validation = dataset(merge(TwoDaySuite, SixDaySuite), p; relative_weights = relative_weight_options["all_but_e"], grid_type=ZGrid, grid_size=64, Δt=10.0);
ce = CalibrationExperiment(calibration, validation, p);

loss = ce.calibration.loss
loss_validation = ce.validation.loss
initial_parameters = ce.calibration.default_parameters
parameternames = propertynames(initial_parameters)

plot_stds_within_bounds(loss, loss_validation, initial_parameters, directory, xrange=-3:0.25:5)
v = plot_prior_variance(loss, loss_validation, initial_parameters, directory; xrange=0.1:0.05:1.0)
plot_num_ensemble_members(loss, loss_validation, initial_parameters, directory; xrange=1:5:30)
nl = plot_observation_noise_level(loss, loss_validation, initial_parameters, directory; xrange=-2.0:0.1:3.0)

