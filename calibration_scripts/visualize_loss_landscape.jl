## Optimizing TKE parameters
using Distributions
using LinearAlgebra
using Random
using EnsembleKalmanProcesses.EnsembleKalmanProcessModule
using EnsembleKalmanProcesses.ParameterDistributionStorage

using Plotly
include("quick_setup.jl")
directory = "LossLandscape/"
isdir(directory) || mkdir(directory)

option = "all_T"
name = L"[\varpi_U, \varpi_V, \varpi_T, \varpi_E] = [1,1,1,1]"
calibration = dataset(FourDaySuite, p; relative_weights = relative_weight_options[option]);

nll = calibration.nll
# nll_validation = ce.validation.nll
initial_parameters = calibration.default_parameters

pvalues = Dict(
        :Cᴷe => collect(0.01:0.01:1.0),
        :Cᴷe => collect(0.001:0.00199:0.20),
        :Cᴰ => collect(0.1:0.1:1.5),
        :Cᴸᵇ => collect(0.1:0.1:5.0),
        :CʷwΔ => collect(0.5:0.5:10.0),
        :Cᴷc => collect(0.01:0.01:0.2),
        :Cᴷu => collect(0.1:0.1:0.5),
        :Cᴬc => collect(1.0:0.1:5.0),
        :Cᴬu => collect(0.001:0.001:0.05),
        :Cᴬe => collect(0.1:0.05:2.0),
        :Cʷu★ => collect(1.0:1.0:10.0),
)

using Plots; pyplot();
xc = []
yc = []
zc = []
for i in 1:length(pvalues[:Cᴷe]), j in 1:length(pvalues[:Cᴬe])
        initial_parameters = TKECalibration2021.custom_defaults(ce.calibration.nll.batch[1].model, ce.parameters.RelevantParameters)
        Cᴷe = pvalues[:Cᴷe][i]
        Cᴬe = pvalues[:Cᴬe][j]
        initial_parameters.Cᴷe = Cᴷe
        initial_parameters.Cᴬe = Cᴬe
        push!(xc, Cᴷe)
        push!(yc, Cᴬe)
        push!(zc, nll(initial_parameters))
end
myplot = Plots.plot(xc,yc,zc,st=:surface,camera=(-45,20))
# p = Plots.plot(xc,yc,zc,st=:surface,camera=(-45,30))
plot!(xlabel=parameter_latex_guide[:Cᴷe], ylabel=parameter_latex_guide[:Cᴬe], colorbar=false, title=name, zlabel=L"\mathcal{L}(\theta)", linewidth=0)
Plots.savefig("loss_landscape_$(option).pdf")
println(xc[argmin(zc)], ", ", yc[argmin(zc)])

# UV: 0.039, 0.1
# T : 0.075, 2.0
# E : 0.11, 0.55

ce = CalibrationExperiment(calibration, validation, p)
initial_parameters = TKECalibration2021.custom_defaults(ce.calibration.nll.batch[1].model, ce.parameters.RelevantParameters)
initial_parameters.Cᴷe = 0.11
initial_parameters.Cᴬe = 0.55
visualize_and_save(ce, initial_parameters, "all_E")

# z = zeros((10,10))
# propertynames(initial_parameters)
# [x for x in 0.01:0.01:1.0]+[x for x in 0.01:0.005:1.0]
#
#
# for pname in keys(pvalues)
#         defaults = TKECalibration2021.custom_defaults(ce.calibration.nll.batch[1].model, ce.parameters.RelevantParameters)
#
#         losses = Dict()
#         for pvalue in pvalues[pname]
#                 TKECalibration2021.set_if_present!(defaults, pname, pvalue)
#                 losses[pvalue] = nll(defaults)
#         end
#         Plots.plot(losses, size=(600,200), lw=4)
#         Plots.savefig("$(pname).png")
# end
#
# losses = Dict()
# for Cᴷe in Cᴷes
#         initial_parameters.Cᴷe = Cᴷe
#         losses[Cᴷe] = nll(initial_parameters)
# end
# Plots.plot(losses)
#
#
# ## GR
# initial_parameters = ce.calibration.default_parameters
# z = zeros((length(pvalues[:Cᴷe]), length(pvalues[:Cᴬe])))
# for i in 1:length(pvalues[:Cᴷe]), j in 1:length(pvalues[:Cᴬe])
#         initial_parameters = TKECalibration2021.custom_defaults(ce.calibration.nll.batch[1].model, ce.parameters.RelevantParameters)
#         Cᴷe = pvalues[:Cᴷe][i]
#         Cᴬe = pvalues[:Cᴬe][j]
#         initial_parameters.Cᴷe = Cᴷe
#         initial_parameters.Cᴬe = Cᴬe
#         z[i,j] = nll(initial_parameters)
# end
# using GR
# GR.surface(z)

## Pyplot
# using Plots; pyplot();
# xc = []
# yc = []
# zc = []
# for i in 1:length(pvalues[:Cᴷe]), j in 1:length(pvalues[:Cᴬe])
#         initial_parameters = TKECalibration2021.custom_defaults(ce.calibration.nll.batch[1].model, ce.parameters.RelevantParameters)
#         Cᴷe = pvalues[:Cᴷe][i]
#         Cᴬe = pvalues[:Cᴬe][j]
#         initial_parameters.Cᴷe = Cᴷe
#         initial_parameters.Cᴬe = Cᴬe
#         push!(xc, Cᴷe)
#         push!(yc, Cᴬe)
#         push!(zc, nll(initial_parameters))
# end
# p = Plots.plot(xc,yc,zc,st=:surface,camera=(-45,20))
# plot!(xlabel=parameter_latex_guide[:Cᴷe], ylabel=parameter_latex_guide[:Cᴬe], title="Loss Landscape", zlabel=L"\mathcal{L}(\theta)", linewidth=0)
# Plots.savefig("loss_landscape5.pdf")
#
# using GLMakie
# surface(xs, ys, zs)
#
#
#
#
# z = []
# for i in 1:length(pvalues[:Cᴷe])
#         Cᴷe = pvalues[:Cᴷe][i]
#         initial_parameters.Cᴷe = Cᴷe
#
#         losses = []
#         for j in 1:length(pvalues[:Cᴬe])
#                 Cᴬe = pvalues[:Cᴬe][j]
#                 initial_parameters.Cᴬe = Cᴬe
#                 push!(losses, nll(initial_parameters))
#         end
#
#         push!(z, losses)
# end
#
# trace = Plotly.surface(z=z)
# layout = Plotly.Layout(title="Mt. Bruno Elevation", autosize=false, width=500,
#                 height=500, margin=attr(l=65, r=50, b=65, t=90))
# p = Plotly.plot(trace, layout)
#
#
#
# p = plot_surface(Cᴷes, Cᴰs, z, rstride=2,edgecolors="k", cstride=2,
#    cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
#
# p.show()
#
# contour(xgrid, ygrid, z, colors="black", linewidth=2.0)
# ax.label(cp, inline=1, fontsize=10)
