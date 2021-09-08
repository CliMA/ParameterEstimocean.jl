
using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.Models.CATKEVerticalDiffusivityModel
using Oceananigans.Fields: interior
using CairoMakie
using LaTeXStrings
using CairoMakie: Figure

LESdata = FourDaySuite
all_fields = [:u, :v, :b, :e]

field_guide = Dict(
    :u => (
        axis_args = (yaxisposition = :left, ylabel="z (m)", xlabel="U velocity (dm/s)"),
        remove_spines = (:t, :r),
        scaling = 1e1,
    ),

    :v => (
        axis_args = (xlabel="V velocity (dm/s)",),
        remove_spines = (:t, :l, :r),
        scaling = 1e1,
    ),

    :b => (
        axis_args = (xlabel="Buoyancy (cN/kg)",),
        remove_spines = (:t, :l, :r),
        scaling = 1e2,
    ),

    :e => (
        axis_args = (ylabel="z (m)", xlabel="TKE (cm²/s²)", ylabelposition=:right, yaxisposition=:right),
        remove_spines = (:t, :l),
        scaling = 1e4,
    )
)


function tostring(num)
    num == 0 && return "0"
    om = Int(floor(log10(abs(num))))
    num /= 10.0^om
    num = num%1 ≈ 0 ? Int(num) : round(num; digits=2)
    return "$(num)e$om"
end

function empty_plot!(fig_position)
    ax = fig_position = Axis(fig)
    hidedecorations!(ax)
    hidespines!(ax, :t, :b, :l, :r)
end

fig = Figure(resolution = (200*(length(all_fields)+1), 200*length(LESdata)), font = "CMU Serif")
colors = [:black, :red, :blue]

for (i, LEScase) in enumerate(values(LESdata))

    td = OneDimensionalTimeSeries(LEScase.filename; grid_type=ColumnEnsembleGrid, Nz=64);
    targets = LEScase.first:(isnothing(LEScase.last) ? length(td.t) : LEScase.last)

    snapshots = round.(Int, range(targets[1], targets[end], length=3))

    empty_plot!(fig[i,1])
    bcs = td.boundary_conditions
    text!("Qᵇ = $(tostring(bcs.Qᵇ)) N kg⁻¹m⁻²s⁻¹\nQᵘ = $(tostring(bcs.Qᵘ)) m⁻¹s⁻²\nf = $(tostring(td.constants[:f])) s⁻¹", 
                position = (0, 0), 
                align = (:center, :center), 
                textsize = 15,
                justification = :left)

    for (j, field) in enumerate([:u, :v, :b, :e])

        j += 1 # reserve the first column for row labels

        info = field_guide[field]

        fdata = getproperty(td, field)

        z = field ∈ [:u, :v] ? td.grid.zF[1:td.grid.Nz] : td.grid.zC[1:td.grid.Nz]

        to_plot = field ∈ relevant_fields(LEScase)

        if to_plot

            ax = to_plot ? Axis(fig[i,j]; xlabelpadding=0, xtickalign=1, ytickalign=1,
                                        info.axis_args...) : 
                                        Axis(fig[i,j]);

            hidespines!(ax, info.remove_spines...)

            field ∈ [:v, :b] && hideydecorations!(ax, grid=false)

            lins = []
            for (color_index, target) in enumerate(snapshots)
                l = lines!([interior(fdata[target]) .* info.scaling ...], z; color = (colors[color_index], 0.4))
                push!(lins, l)
            end
            times = @. round((td.t[snapshots] - td.t[snapshots[1]]) / 86400, sigdigits=2)
            Legend(fig[1,2:3], lins, ["LES, t = $time days" for time in times])
            lins = []
        else
            
            empty_plot!(fig[i,j])
        end

    end

    save("filename.png", fig, px_per_unit = 2.0) 
    display(fig);

end
