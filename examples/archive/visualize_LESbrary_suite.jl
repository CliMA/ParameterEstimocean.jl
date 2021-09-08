using Plots
using TKECalibration2021
using OceanTurbulenceParameterEstimation

data_path = "/Users/adelinehillier/.julia/dev/6DaySuite/"

files =  ["free_convection",
          "strong_wind",
          "strong_wind_no_rotation",
          "weak_wind_strong_cooling",
          "strong_wind_weak_cooling",
          ]

output_directory=data_path*"/ProfileVisuals/"
mkpath(output_directory)

Ts = Dict()
for file in files
    Ts[file] = OneDimensionalTimeSeries(data_path*file*"/instantaneous_statistics.jld2")
end

file_labels = Dict(
    "free_convection" => "Free convection",
    "strong_wind" => "Strong wind",
    "strong_wind_no_rotation" => "Strong wind, no rotation",
    "weak_wind_strong_cooling" => "Weak wind, strong cooling",
    "strong_wind_weak_cooling" => "Strong wind, weak cooling",
    "strong_wind_weak_heating" => "Strong wind, weak heating"
)

f = Dict(
    "u" => 𝒟 -> 𝒟.U,
    "v" => 𝒟 -> 𝒟.V,
    "e" => 𝒟 -> 𝒟.e,
    "uw" => 𝒟 -> 𝒟.UW,
    "vw" => 𝒟 -> 𝒟.VW,
    "wT" => 𝒟 -> 𝒟.wT,
    "T"  => 𝒟 -> 𝒟.T
)

legend_placement = Dict(
    "u" => false,
    "v" => false,
    "e" => false,
    "T" => false,
    "uw" => false,
    "vw" => false,
    "wT" => false,
    "T"  => false,
)

scaling_factor = Dict(
    "uw" => 1e4,
    "vw" => 1e4,
    "wT" => 1e4,
    "e" => 1e3,
    "u" => 1,
    "v" => 1,
    "T" => 1
)

x_labels = Dict(
    "u" => "U (m/s)",
    "v" => "V (m/s)",
    "uw" => "U'W' x 10⁴ (m²/s²)",
    "vw" => "V'W' x 10⁴ (m²/s²)",
    "wT" => "W'T' x 10⁴ (C⋅m/s)",
    "T" => "T (C)",
    "e" => "E x 10³ (m²/s²)",
)

titles = Dict(
    "uw" => "Zonal momentum flux, U'W'",
    "vw" => "Meridional momentum flux, V'W'",
    "wT" => "Temperature flux, W'T'",
    "u" => "East-West Velocity, U",
    "v" => "North-South Velocity, V",
    "T" => "Temperature, T",
    "e" => "Turbulent Kinetic Energy, E",
)


function plot_final_frame(name)
    p = Plots.plot(xlabel=x_labels[name], ylabel="Depth (m)", palette=:darkrainbow, legend=legend_placement[name], foreground_color_grid=:white, plot_titlefontsize=20)
    for (file, X) in Ts
        cell_field = f[name](X) # f["T"](Ts["free_convection"])
        Nz = cell_field[1].grid.N
        z = parent(cell_field[1].grid.zc[1:Nz-1])
        profile = cell_field[end].data[1:Nz-1]
        plot!(profile.*scaling_factor[name], z, title = titles[name], label="$(file)", linewidth=4, la=0.5)
    end
    plot!(size=(400,500))
    p
end

for name in ["u", "v", "T", "e"]
    p = plot_final_frame(name)
    Plots.savefig(p, output_directory*"/$(name)_last_frame.pdf")
end

p = Plots.plot(grid=false, showaxis=false, palette=:darkrainbow, ticks=nothing)
for (file, T) in Ts
    Plots.plot!(1, label=file_labels[file], legend=:left, size=(200,600))
end
p
Plots.savefig(p, output_directory*"/legend_last_frame.pdf")

# layout = @layout [a b c d]
# p = Plots.plot(p1,p2,p3,p4,layout=layout, size=(1600,400), tickfontsize=12)
# savefig(p, output_directory*"/all_last_frame_new_suite.pdf")
#
# layout = @layout [a b c d e]
# p = Plots.plot(p1,p2,p3,pT,p4,layout=layout, size=(1600,400), tickfontsize=12)
# savefig(p, output_directory*"/all_last_frame_new_suite_with_T.pdf")

cell_field = f["T"](Ts["free_convection"])
Nt = length(cell_field)
Nz = cell_field[1].grid.N
z = parent(cell_field[1].grid.zc[1:Nz])
function animate_(name)
    anim = @animate for i=1:Nt
        p = Plots.plot(xlabel=x_labels[name], ylabel="Depth (m)", palette=:darkrainbow, legend=legend_placement[name], foreground_color_grid=:white, plot_titlefontsize=20)
        for (file, X) in Ts
            cell_field = f[name](X) # f["T"](Ts["free_convection"])
            profile = cell_field[i].data[1:Nz]
            plot!(profile.*scaling_factor[name], z[1:Nz], title = titles[name], label="$(file)", linewidth=4, la=0.5)
        end
        plot!(size=(400,500))
        p
    end
    return anim
end

for name in ["u", "v", "T", "e"]
    anim = animate_(name)
    gif(anim, output_directory*"/$(name).gif", fps=20)
end

##

x_lims = Dict(
    "u" => (-0.15,0.4),
    "v" => (-0.3,0.1),
    "T" => (19.6,20.0),
    "e" => (-0.5,4),
)

cell_field = f["T"](Ts["free_convection"])
Nt = length(cell_field)
Nz = cell_field[1].grid.N
z = parent(cell_field[1].grid.zc[1:Nz])
# X = Ts["strong_wind_weak_cooling"]
function animate_(name)
    # cell_field = f[name](X) # f["T"](Ts["free_convection"])
    anim = @animate for i=12:Nt
        p = Plots.plot(xlabel=x_labels[name], ylabel="Depth (m)", palette=:darkrainbow, legend=false, foreground_color_grid=:white, plot_titlefontsize=20)
        for (file, X) in Ts
            cell_field = f[name](X)
            profile = cell_field[i].data[1:Nz-1]
            plot!(profile.*scaling_factor[name], z[1:Nz-1], title = titles[name], linewidth=5, xlims = x_lims[name])
            plot!(size=(400,500))
        end
    end
    return anim
end

for name in ["u", "v", "T", "e"]
    anim = animate_(name)
    gif(anim, output_directory*"/$(name).gif", fps=40)
end
