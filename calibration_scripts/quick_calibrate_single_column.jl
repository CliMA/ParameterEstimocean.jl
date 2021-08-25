using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.CATKEVerticalDiffusivityModel
using OceanTurbulenceParameterEstimation.ModelsAndData
using OceanTurbulenceParameterEstimation.ParameterEstimation
using Statistics, Plots, Dao, PyPlot, Dates

relative_weight_option = "all_but_e"
free_parameter_option = "TKEParametersRiDependent"

p = Parameters(RelevantParameters = free_parameter_options[free_parameter_option],
               ParametersToOptimize = free_parameter_options[free_parameter_option])

calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options["all_but_e"],
                                        grid_type=ColumnEnsembleGrid,
                                        Nz=64,
                                        Δt=10.0);

validation = dataset(merge(TwoDaySuite, SixDaySuite), p;
                                        relative_weights = relative_weight_options["all_but_e"],
                                        grid_type=ColumnEnsembleGrid,
                                        Nz=64,
                                        Δt=10.0);

ce = CalibrationExperiment(calibration, validation, p);

output = model_time_series(ce.default_parameters, ce.calibration.model, ce.calibration.data_batch[1])

# visualize_and_save(ce, ce.default_parameters, pwd())
