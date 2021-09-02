using TKECalibration2021
using StatsPlots
using Distributions
using LinearAlgebra
using Random
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage
using ArgParse
using OceanTurbulenceParameterEstimation
using Plots
using Dao
using PyPlot

# using StatsPlots
# pl = StatsPlots.plot(label=0.4, xlims=(0.0,4), title="Parameter Priors")
# for x in 0.3:0.3:1.5
#     StatsPlots.plot!(LogNormal(0.0,x), label="$(x)", lw=4, la=0.7, size=(375,250), palette = :darkrainbow, legendtitle=L"\Gamma_\theta", xlabel=L"\theta^{(i)}", ylabel=L"p_{prior}(\theta^{(i)})")
# end
# plot!(annotation=(1.9,1.0,L"lnN(0, \Gamma_\theta)"))
# pl
# StatsPlots.savefig("priors.pdf")

# s = ArgParseSettings()
#
# @add_arg_table! s begin
#     "--relative_weight_option"
#         help = ""
#         default = "all_but_e"
#         arg_type = String
# end
# relative_weight_option = parse_args(s)["relative_weight_option"]
#
# directory = "Distributions/TKEParametersConvectiveAdjustmentRiDependent/"
# isdir(directory) || mkdir(directory)
#
# # @free_parameters(ConvectiveAdjustmentParameters,
# #                  Cᴬu, Cᴬc, Cᴬe)
#
# relative_weight_options = Dict(
#                 "all_e" => Dict(:T => 0.0, :U => 0.0, :V => 0.0, :e => 1.0),
#                 "all_T" => Dict(:T => 1.0, :U => 0.0, :V => 0.0, :e => 0.0),
#                 "uniform" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 1.0),
#                 "all_but_e" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 0.0),
#                 "all_uv" => Dict(:T => 0.0, :U => 1.0, :V => 1.0, :e => 0.0)
# )
#
# p = Parameters(RelevantParameters = TKEParametersConvectiveAdjustmentRiIndependent,
#                ParametersToOptimize = TKEParametersConvectiveAdjustmentRiIndependent
#               )
#
# # relative_weight_option = "uniform"
# calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options[relative_weight_option]);
# validation = dataset(merge(TwoDaySuite, SixDaySuite), p; relative_weights = relative_weight_options[relative_weight_option]);
# ce = CalibrationExperiment(calibration, validation, p)
#
# loss = ce.calibration.loss
# loss_validation = ce.validation.loss
# initial_parameters = ce.calibration.default_parameters

# loss_validation([initial_parameters...])

ce.parameters.RelevantParameters([initial_parameters...])
propertynames(initial_parameters)

function get_losses(pvalues, pname, loss)
    defaults = TKECalibration2021.custom_defaults(ce.calibration.loss.batch[1].model, ce.parameters.RelevantParameters)
    ℒvalues = []
    for pvalue in pvalues
        TKECalibration2021.set_if_present!(defaults, pname, pvalue)
        # println(defaults)
        push!(ℒvalues, loss([defaults...]))
    end
    return ℒvalues
end


# pname = :Cᴬu
# pvalues = range(0.0, stop=3.0, length=1000)
# pvalues = range(0.001, stop=0.1, length=99)
# loss = loss
# a = get_losses(pvalues, pname, loss)
# pvalues[argmin(a)]
# Plots.plot(pvalues, a)

function lognormal_μ_σ²(mean, variance)
    k = variance / mean^2 + 1
    μ = log(mean / sqrt(k))
    σ² = log(k)
    return μ, σ²
end

function get_μ_σ²(mean, variance, bounds)
    if bounds[1] == 0.0
        return lognormal_μ_σ²(mean, variance)
    end
    return mean, variance
end

function get_constraint(bounds)
    if bounds[1] == 0.0
        return bounded_below(0.0)
    end
    return no_constraint()
end

initial_parameters = ParametersToOptimize(initial_parameters)
bounds, prior_variances = get_bounds_and_variance(initial_parameters; stds_within_bounds = 3);
prior_means = [initial_parameters...]
# μs, σ²s = lognormal_μ_σ²(prior_means, prior_variances)


μs = [get_μ_σ²(prior_means[i], prior_variances[i], bounds[i])[1] for i in eachindex(prior_means)]
σ²s = [get_μ_σ²(prior_means[i], prior_variances[i], bounds[i])[2] for i in eachindex(prior_means)]


# first term (data misfit) of EKI objective = ℒ / obs_noise_level
ℒ = ce.calibration.loss(prior_means)
println(ℒ)

# second term (prior misfit) of EKI objective = || σ²s.^(-0.5) .* μs ||²
pr = norm((σ²s.^(-1/2)) .* μs)^2
println(pr)

# for equal weighting of data misfit and prior misfit in EKI objective, let obs noise level be about
obs_noise_level = ℒ / pr

# obs = 1e-3
# ℒ / (obs*pr) = 100
# obs = ℒ / (100 * pr)

all_plots = []
for i in 1:length(initial_parameters)
    pname = propertynames(initial_parameters)[i]
    pvalue_lims = (max(0.0, prior_means[i]-sqrt(prior_variances[i])), prior_means[i]+2*sqrt(prior_variances[i]))
    pvalues = range(pvalue_lims[1],stop=pvalue_lims[2],length=20)
    losses_cal = get_losses(pvalues, pname, loss)
    losses_val = get_losses(pvalues, pname, loss_validation) ./ 2

    kwargs = (lw=4, xlims = pvalue_lims, xlabel = "$(parameter_latex_guide[pname])")
    distplot = StatsPlots.plot(LogNormal(μs[i], σ²s[i]); ylabel = L"P_{prior}(\theta)", label="", color=:red, kwargs...)
    plot!([prior_means[i]], linetype = :vline, linestyle=:dash, color=:red, label="mean")

    lossplot = Plots.plot(pvalues, log.(losses_val), label="val", color=:blue, lw=4, la=0.5)
    plot!(pvalues, log.(losses_cal);  xlabel="$(parameter_latex_guide[pname])", color=:purple, la=0.5, label="cal", ylabel = L"\log_{10}\mathcal{L}(\theta)", kwargs...)
    plot!([pvalues[argmin(losses_cal)]], linetype = :vline, linestyle=:dash, color=:purple, la=0.5, label="min")
    plot!([pvalues[argmin(losses_val)]], linetype = :vline, linestyle=:dash, color=:blue, la=0.5, label="min")

    layout = @layout [a;b]
    pplot = Plots.plot(distplot, lossplot; layout=layout, framestyle=:box)
    Plots.savefig(directory*"$(pname)_$(relative_weight_option).png")
    push!(all_plots, pplot)
end
Plots.plot(all_plots..., layout=(3,5), size=(1600,1200), left_margin=10*Plots.mm, bottom_margin=10*Plots.mm)
Plots.savefig(directory*"parameters_$(relative_weight_option).pdf")
