using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.ModelsAndData
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

args = (ensemble_size=50, Nz=32, Δt=10.0)

p = Parameters(RelevantParameters = free_parameter_options[free_parameter_option],
               ParametersToOptimize = free_parameter_options[free_parameter_option])

calibration = ensemble_dataset(FourDaySuite, p;
                         relative_weights = relative_weight_options["all_but_e"],
                             args...)

validation = ensemble_dataset(merge(TwoDaySuite, SixDaySuite), p;
                         relative_weights = relative_weight_options["all_but_e"],
                             args...)

ce = CalibrationExperiment(calibration, validation, p);

directory = "EKI/$(free_parameter_option)_$(relative_weight_option)/"
isdir(directory) || mkpath(directory)

loss = ce.calibration.loss
# loss_validation = ce.validation.loss
initial_parameters = ce.default_parameters
parameternames = propertynames(initial_parameters)

loss(calibration.default_parameters)