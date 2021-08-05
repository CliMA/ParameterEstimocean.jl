using OceanTurbulenceParameterEstimation
using TKECalibration2021
include("../../src/setup.jl")
include("../../src/utils.jl")

casename = "free_convection"
relative_weights = [1e+0, 1e-4, 1e-4, 1e-4]
LEScase = FourDaySuite[casename]
datapath = joinpath(FourDaySuite_path, LEScase.filename)

RelevantParameters = TKEConvectiveAdjustmentRiIndependent
ParametersToOptimize = TKEConvectiveAdjustmentRiIndependent

nll = init_tke_calibration(datapath;
                                         N = 32,
                                        Δt = 60.0, #1minute
                              first_target = LEScase.first,
                               last_target = LEScase.last,
                                    fields = tke_fields(LEScase),
                          relative_weights = relative_weights,
                        eddy_diffusivities = TKEMassFlux.IndependentDiffusivities(),
                     convective_adjustment = TKEMassFlux.VariablePrandtlConvectiveAdjustment(),
                        )

model = nll.model
tdata = nll.data
set!(model, custom_defaults(model, RelevantParameters))
initial_parameters = custom_defaults(model, ParametersToOptimize)

# Run the case
calibration = calibrate(nll, initial_parameters, samples = 1000, iterations = 10)

# Save results
@save results calibration

# Do some simple analysis
 loss = calibration.negative_log_likelihood.loss
chain = calibration.markov_chains[end]
   C★ = optimal(chain).param

close("all")
viz_fig, viz_axs = visualize_realizations(model, data, loss.targets[[1, end]], C★,
                                           fields = (:T, :e),
                                          figsize = (16, 6))
