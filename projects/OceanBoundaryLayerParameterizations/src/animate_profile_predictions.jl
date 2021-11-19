
colors = Dict("u" => :blue, "v" => :green, "b" => :red, :e => :yellow)
x_lims = Dict("u" => (-0.3,0.4), "v" => (-0.3,0.1), "b" => (19.6,20.0), "e" => (-0.5,4))

function animate_LESbrary_suite(ce, directory; parameters=ce.default_parameters, targets=ce.validation.loss.loss.batch[1].loss.targets)

    truth = Dict()
    model = Dict()
    for myloss in ce.validation.loss.loss.batch
        truth[myloss.data.name] = myloss.data
        model[myloss.data.name] = model_time_series(parameters, myloss.model, myloss.data)
    end

    function plot_(name, file, t)
        start=13
        truth_field = getproperty(truth[file], Symbol(name))
        catke_field = getproperty(model[file], Symbol(name))
        Nz = truth_field[1].grid.Nz
        z = parent(truth_field[1].grid.zᵃᵃᶜ[1:Nz-1])
        Nz_catke = catke_field[1].grid.Nz
        z_catke = parent(catke_field[1].grid.zᵃᵃᶜ)[1:Nz_catke-1]
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
                truth_profile = getproperty(truth[file], :v)[t+13].data[1:Nz-1]
                catke_profile = getproperty(model[file], :v)[t].data[1:Nz_catke-1]
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
        t = plot_("b", file, t)
        layout=@layout[a; b]
        p = Plots.plot(t, u, layout=layout)
        plot!(tickfontsize=20, ylims=(-256,0), ticks=false)
        plot!(widen=true, grid=false, framestyle=:none)
        return p
    end

    function LESbrary_suite_snapshot(t)
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

    anim = @animate for t=targets
        p = LESbrary_suite_snapshot(t)
    end

    Plots.gif(anim, directory, fps=400)
end
