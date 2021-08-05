using Plots
using TKECalibration2021
using OceanTurbulenceParameterEstimation

files =  ["free_convection",
          "strong_wind",
          "strong_wind_no_rotation",
          "weak_wind_strong_cooling",
          "strong_wind_weak_cooling",
          ]

# TKEParametersRiDependent
# params = [1.8499517368851253, 10.744045010185022, 0.2552785943217983, 0.48295965480135017, 0.08997933917838298, 0.060700175544787675, 1.5830296122806629, 52.945204927578956, 4.2436100116907625, 2.251821910327211, 5.713895379598551, 0.933486665976551]
# params = [0.7212840482250865, 0.758750120145174, 0.1513196166076823, 0.7337339504959434, 0.39766166345769866, 1.773824568337, 0.13342507002807857, 1.2247403461138953, 2.907953616918013, 1.1612258856834292, 3.618765767212402, 1.3051900286568907]
# params = [1.0107465739445465, 1167.4226071563912, 0.17847471335704368, 1.4143699084523462, 0.5166096177740676, 0.42893520907485144, 0.10577255139676721, 0.40691015103153355, 1.5474819461430098, 0.8184388831898733, 2.2599104304549114, 0.7549306030957961]

# TKEParametersConvectiveAdjustmentRiIndependent
# params = [0.5129935182296373, 1.4251154326181028, 0.9506909837492364, 1.3949910825036915, 0.3146093521875936, 1.105856173400302, 1.7119197469814198, 1.224520121155424, 0.9606209174282703, 0.4546079775804564, 1.0359089052988595, 0.9057416995529349, 0.7913567861402568, 0.7400590096732186, 1.3077769117076257]
# params = [0.12702071444936408, 84.30661835279565, 0.45342284361367813, 0.2546732758792878, 5.483368782697504, 0.011870875508671832, 0.025026596750480464, 61.25348831546525, 0.0448409350242226, 0.00017033850219665893, 0.0029349769160469377, 0.14262715122871128, 0.000572041621587704, 0.18699815008636606, 0.0008096603432199394]

# new defaults --> TKEParametersConvectiveAdjustmentRiDependent
# params = [1.01, 1167.0, 0.1784, 1.4144, 0.5166, 0.4289, 0.10577, 0.4069, 1.5475, 0.8184, 2.2599, 0.7549, 0.0057, 0.6706, 0.2717]

# old defaults --> TKEParametersConvectiveAdjustmentRiIndependent
# params = [0.1243, 0.09518, 0.1243, 0.5915, 1.1745, 6.645, 0.0579, 0.6706, 0.0057, 0.2717]
propertynames(initial_parameters)

# set_if_present!(defaults, :Cᴷc, 0.09518)
# set_if_present!(defaults, :Cᴷe, 0.055)
# set_if_present!(defaults, :Cᴷu, 0.1243)
# set_if_present!(defaults, :Cᴰ,  0.5915)
# set_if_present!(defaults, :Cᴸᵇ, 1.1745)
# set_if_present!(defaults, :Cʷu★, 6.645)
# set_if_present!(defaults, :CʷwΔ, 0.0579)
# set_if_present!(defaults, :Cᴬc, 0.6706)
# set_if_present!(defaults, :Cᴬu, 0.0057)
# set_if_present!(defaults, :Cᴬe, 0.2717)

# # Independent diffusivities
#
# # Convective Adjustment
# set_if_present!(defaults, :Cᴬc, 0.6706)
# set_if_present!(defaults, :Cᴬu, 0.0057)
# set_if_present!(defaults, :Cᴬe, 0.2717)
#
# set_if_present!(defaults, :CᴷRiʷ, 0.5)
# set_if_present!(defaults, :CᴷRiᶜ, -0.73)
# set_if_present!(defaults, :Cᴷu⁻, 0.7062)
# set_if_present!(defaults, :Cᴷu⁺, 0.0071)
# set_if_present!(defaults, :Cᴷc⁻, 4.5015)
# set_if_present!(defaults, :Cᴷe⁻, 1.2482)
# set_if_present!(defaults, :Cᴷe⁺, 0.0323)


params = [0.7212840482250865, 0.758750120145174, 0.1513196166076823, 0.7337339504959434, 0.39766166345769866, 1.773824568337, 0.13342507002807857, 1.2247403461138953, 2.907953616918013, 1.1612258856834292, 3.618765767212402, 1.3051900286568907]

# Convective Adjustment

params = [19.856400080198597, 4.457257067068098]

# ce.validation.nll(initial_parameters)
# ce.validation.nll_wrapper(params)
# params = initial_parameters

params = [3.618765767212402, 1.3051900286568907]
ce.calibration.nll_wrapper(params)
ce.calibration.nll_wrapper([1.3169453047222575, 1.5077502014763762])
ce.calibration.nll_wrapper([19.856400080198597, 4.457257067068098])

parameters = ce.default_parameters
parameters = ce.parameters.ParametersToOptimize(params)
Truth = Dict()
CATKE = Dict()
for mynll in ce.validation.nll.batch
    Truth[mynll.data.name] = mynll.data
    CATKE[mynll.data.name] = model_time_series(parameters, mynll)
end

f = Dict(
    "u" => 𝒟 -> 𝒟.U,
    "v" => 𝒟 -> 𝒟.V,
    "e" => 𝒟 -> 𝒟.e,
    "T"  => 𝒟 -> 𝒟.T
)
colors = Dict(
    "u" => :blue,
    "v" => :green,
    "T" => :red
)

colors = Dict(
    "u" => :blue,
    "v" => :green,
    "T" => :red
)

x_lims = Dict(
    "u" => (-0.3,0.4),
    "v" => (-0.3,0.1),
    "T" => (19.6,20.0),
    "e" => (-0.5,4),
)

function plot_(name, file, t)
    start=13
    truth_field = f[name](Truth[file])
    catke_field = f[name](CATKE[file])
    Nz = truth_field[1].grid.N
    z = parent(truth_field[1].grid.zc[1:Nz-1])
    Nz_catke = catke_field[1].grid.N
    z_catke = parent(catke_field[1].grid.zc)[1:Nz_catke-1]
    p = Plots.plot(legend=false, plot_titlefontsize=20, xlims=x_lims[name])
    for t = t
        truth_profile = truth_field[t+13].data[1:Nz-1]
        catke_profile = catke_field[t].data[1:Nz_catke-1]
        plot!(truth_profile, z, color=colors[name], linewidth=10, la=0.3)
        if !any(i -> isnan(i), catke_profile)
            plot!(catke_profile, z_catke, color=colors[name], linewidth=3, linestyle=:solid)
            # scatter!(catke_profile, z_catke, markersize=3, color=colors[name], linewidth=10)
        end
        if name=="u"
            truth_profile = f["v"](Truth[file])[t+13].data[1:Nz-1]
            catke_profile = f["v"](CATKE[file])[t].data[1:Nz_catke-1]
            plot!(truth_profile, z, color=colors["v"], linewidth=10, la=0.3)
            if !any(i -> isnan(i), catke_profile)
                plot!(catke_profile, z_catke, color=colors["v"], linewidth=3, linestyle=:solid)
                # scatter!(catke_profile, z_catke, markersize=3, color=colors["v"], linewidth=10)
            end
        end
    end
    p
end

function stacked_(file, t)
    u = plot_("u", file, t)
    # v = plot_("v", file)
    t = plot_("T", file, t)
    layout=@layout[a; b]
    p = Plots.plot(t, u, layout=layout)
    plot!(tickfontsize=20, ylims=(-256,0), ticks=false)
    plot!(widen=true, grid=false, framestyle=:none)
    # plot!(background_color=:Greys_4)
    return p
end

function whole_suite_animation(t)
    a = stacked_("free_convection", t)
    b = stacked_("strong_wind", t)
    c = stacked_("strong_wind_no_rotation", t)
    d = stacked_("weak_wind_strong_cooling", t)
    e = stacked_("strong_wind_weak_cooling", t)
    layout = @layout [a b c d e]
    p = Plots.plot(a, b, c, d, e, layout=layout, framestyle=:none)
    plot!(bottom_margin=0*Plots.mm, size=(1800, 800))
    return p
end

whole_suite_animation(840)

anim = @animate for t=1:2:840
    p = whole_suite_animation(t)
end

Plots.gif(anim, "./$(free_parameter_type)_64_calibrated.gif", fps=400)

visualize_and_save(ce, params, "hello/")
# function plot_(name)
#     cell_field = f[name](T)
#     Nz = cell_field[1].grid.N
#     z = parent(cell_field[1].grid.zc[1:Nz-1])
#     starting_profile = cell_field[1].data[1:Nz-1]
#     p = Plots.plot(legend=false, plot_titlefontsize=20)
#     plot!(starting_profile.*scaling_factor[name], z, color=:purple, label="Initial condition (t=0)", linewidth=6, linestyle=:dot)
#     for t = [0.5, 6]
#         profile = cell_field[Int(t*144)].data[1:Nz-1]
#         plot!(profile.*scaling_factor[name], z, label="t = $(t) days", linewidth=8, palette=:default)
#     end
#     p
# end
# u = plot_("u")
# v = plot_("v")
# t = plot_("T")
# layout=@layout[a b c]
# p = Plots.plot(u, v, t, layout=layout)
# plot!(size=(1200,500), tickfontsize=20, ylims=(-256,0), ticks=false)
# plot!(margin=10*Plots.mm, widen=true, grid=false, framestyle=:none)
# plot!(background_color=:Greys_4)
# Plots.savefig(p, output_directory*"/$(file).pdf")
