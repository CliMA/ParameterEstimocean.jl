using TKECalibration2021

LESdata = merge(FourDaySuite, GeneralStrat)
weights = [1.0 for d in LESdata]

RelevantParameters = TKEParametersConvectiveAdjustmentRiIndependent
ParametersToOptimize = TKEParametersConvectiveAdjustmentRiIndependent
dname = "calibrate_FourDaySuiteGeneralStrat"
# RiIndependentTKEParameters
# RiIndependentTKEParametersConvectiveAdjustment
# TKEFreeConvectionConvectiveAdjustmentRiIndependent
# TKEFreeConvectionRiIndependent

nll, initial_parameters = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize)

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

writeout2(o, "Default", initial_parameters, nll(initial_parameters))

@info "Running Iterative Simulated Annealing..."

global initial_parameters

calibration = simulated_annealing(nll, initial_parameters; samples=1000, iterations=10)

savename = @sprintf("tke_batch_calibration_convection_refine_dz%d_dt%d_2.jld2", batched_nll.batch[1].model.grid.Δc,
                    batched_nll.batch[1].model.Δt / minute)

@save savename calibration

println("done")

myfilename = String(savename)

global myfilename

include("mega_batch_visualization.jl")

opt = optimal(calibration.markov_chains[end])
optimal_parameters = opt.param
writeout2(o, "Annealing", optimal_parameters, nll(params))
