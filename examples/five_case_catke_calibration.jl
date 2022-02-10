using Oceananigans
using Oceananigans.Units
using OceanTurbulenceParameterEstimation
using LinearAlgebra, CairoMakie, DataDeps, Distributions

using ElectronDisplay

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    SurfaceTKEFlux,
    MixingLength

#####
##### Compile LESbrary
#####

case_path(case) = @datadep_str("four_day_suite_4m/$(case)_instantaneous_statistics.jld2")

times = [96hours - 12hour, 96hours]
field_names = (:b, :e, :u, :v)
regrid_size = nothing #(1, 1, 32)

transformation = (b = ZScore(),
                  u = RescaledZScore(1e-2),
                  v = RescaledZScore(1e-2),
                  e = RescaledZScore(1e-2))

observation_library = Dict()

# Don't optimize u, v for free_convection
observation_library["free_convection"] =
    SyntheticObservations(case_path("free_convection"); transformation, times, regrid_size,
                          field_names = (:b, :e))
                                                                
# Don't optimize v for non-rotating cases
observation_library["strong_wind_no_rotation"] =
    SyntheticObservations(case_path("strong_wind_no_rotation"); transformation, times, regrid_size,
                          field_names = (:b, :e, :u))

# The rest are standard
for case in ["strong_wind", "strong_wind_weak_cooling", "weak_wind_strong_cooling"]
    observation_library[case] = SyntheticObservations(case_path(case); field_names, transformation, times, regrid_size)
end

cases = [
         "free_convection",
         "weak_wind_strong_cooling",
         "strong_wind_weak_cooling",
         "strong_wind",
         "strong_wind_no_rotation",
        ]

observations = [observation_library[case] for case in cases]
     
#####
##### Simulation
#####

# Constant Ri, no convection
mixing_length = MixingLength(Cᴬu   = 0.0,
                             Cᴬc   = 0.0,
                             Cᴬe   = 0.0,
                             Cᴸᵇ   = 1.36,
                             Cᴷu⁻  = 0.101,
                             Cᴷc⁻  = 0.0574,
                             Cᴷe⁻  = 3.32,
                             Cᵟu   = 0.5,
                             Cᵟc   = 0.5,
                             Cᵟe   = 0.5,
                             CᴷRiᶜ = 2.0,
                             Cᴷuʳ  = 0.0,
                             Cᴷcʳ  = 0.0,
                             Cᴷeʳ  = 0.0)

surface_TKE_flux = SurfaceTKEFlux(CᵂwΔ=4.74, Cᵂu★=2.76)
catke = CATKEVerticalDiffusivity(; Cᴰ=1.78, mixing_length)

#####
##### Calibration
#####

mass = 0.6
prior_library = Dict()
prior_library[:CᵂwΔ]  = ScaledLogitNormal(; bounds=(0, 12), interval=(2, 5), mass)
prior_library[:Cᵂu★]  = ScaledLogitNormal(; bounds=(0, 11), interval=(3, 5), mass)
prior_library[:Cᴸᵇ]   = ScaledLogitNormal(; bounds=(0,  6), interval=(0.1, 2), mass)
prior_library[:Cᴰ]    = ScaledLogitNormal(; bounds=(0,  6), interval=(0.5, 2), mass)

prior_library[:Cᴷu⁻]  = ScaledLogitNormal(; bounds=(0, 0.3)) #, interval=(0.01, 0.1), mass)
prior_library[:Cᴷc⁻]  = ScaledLogitNormal(; bounds=(0, 0.5)) #, interval=(0.5, 1.0), mass)
prior_library[:Cᴷe⁻]  = ScaledLogitNormal(; bounds=(0, 10)) #, interval=(1.5, 3), mass)

prior_library[:Cᴷuʳ]  = ScaledLogitNormal(; bounds=(0,  3), interval=(2, 2.5), mass)
prior_library[:Cᴷcʳ]  = ScaledLogitNormal(; bounds=(0,  6), interval=(3, 5), mass)
prior_library[:Cᴷeʳ]  = ScaledLogitNormal(; bounds=(0,  1)) #, interval=(0.5, 2), mass)

prior_library[:CᴷRiᶜ] = ScaledLogitNormal(; bounds=(1, 3)) #)
prior_library[:CᴷRiʷ] = ScaledLogitNormal(; bounds=(0, 0.3)) #)

prior_library[:Cᴬu]   = ScaledLogitNormal(; bounds=(0, 1))
prior_library[:Cᴬc]   = ScaledLogitNormal(; bounds=(0, 10), interval=(3, 6), mass)
prior_library[:Cᴬe]   = ScaledLogitNormal(; bounds=(0, 2)) #, interval=(1, 3), mass)

prior_library[:Cᵟu]  = ScaledLogitNormal(; bounds=(0, 10))
prior_library[:Cᵟc]  = ScaledLogitNormal(; bounds=(0, 10))
prior_library[:Cᵟe]  = ScaledLogitNormal(; bounds=(0, 10))

# No convective adjustment:
constant_Ri_parameters = (:Cᴰ, :CᵂwΔ, :Cᵂu★, :Cᴸᵇ, :Cᴷu⁻, :Cᴷc⁻, :Cᴷe⁻, :Cᵟu, :Cᵟc, :Cᵟe)
variable_Ri_parameters = (:Cᴷuʳ, :Cᴷcʳ, :Cᴷeʳ, :CᴷRiʷ, :CᴷRiᶜ, :Cᴰ, :Cᴸᵇ, :CᵂwΔ, :Cᵂu★)
convective_adjustment_parameters = (:Cᴬc, :Cᴬe) # Cᴬu

# For tomorrow
parameter_names = (:CᵂwΔ, :Cᵂu★, :Cᴷe⁻, :Cᴸᵇ, :Cᴰ, :Cᴷc⁻, :Cᴷu⁻, :Cᴷuʳ, :Cᴷcʳ, :Cᴷeʳ, :CᴷRiᶜ, :CᴷRiʷ, :Cᴬc, :Cᴬe)
free_parameters = FreeParameters(prior_library, names=parameter_names)

noise_covariance = 1e2
simulation.Δt = 5.0 

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 1000,
                                              architecture = GPU(),
                                              tracers = (:b, :e),
                                              closure = catke)

Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

for (case, obs) in enumerate(observations)
    @show case cases[case]
    @show obs.metadata.parameters.momentum_flux
    @show obs.metadata.parameters.buoyancy_flux
    @show f = obs.metadata.parameters.coriolis_parameter

    view(Qᵘ, :, case) .= obs.metadata.parameters.momentum_flux
    view(Qᵇ, :, case) .= obs.metadata.parameters.buoyancy_flux
    view(N², :, case) .= obs.metadata.parameters.N²_deep
    view(simulation.model.coriolis, :, case) .= Ref(FPlane(f=f))
end

calibration = InverseProblem(observations, simulation, free_parameters)

eki = EnsembleKalmanInversion(calibration; noise_covariance,
                              resampler = Resampler(acceptable_failure_fraction = 0.5,
                                                    only_failed_particles = true))

#####
##### Plot utils
#####

Nt = length(times)
observed_data = []

for observation in observations
    time_serieses = observation.field_time_serieses
    names = keys(time_serieses)
    case_data = NamedTuple(n => interior(getproperty(time_serieses, n)[Nt])[1, 1, :] for n in names)
    push!(observed_data, case_data)
end

function get_modeled_case(icase, name, k=1)
    model_time_serieses = calibration.time_series_collector.field_time_serieses 
    field = getproperty(model_time_serieses, name)[Nt]
    return view(interior(field), k, icase, :)
end

mean_modeled_data        = [NamedTuple(n => get_modeled_case(c, n, 1)
                                       for n in keys(observations[c].field_time_serieses)) for c = 1:length(observations)]
best_modeled_data        = [NamedTuple(n => get_modeled_case(c, n, 2)
                                       for n in keys(observations[c].field_time_serieses)) for c = 1:length(observations)]
latest_best_modeled_data = [NamedTuple(n => get_modeled_case(c, n, 3)
                                       for n in keys(observations[c].field_time_serieses)) for c = 1:length(observations)]
worst_modeled_data       = [NamedTuple(n => get_modeled_case(c, n, 4)
                                       for n in keys(observations[c].field_time_serieses)) for c = 1:length(observations)]

colorcycle =  [:black, :royalblue1, :darkgreen, :lightsalmon, :seagreen, :magenta2, :red4, :khaki1,   :darkgreen, :bisque4,
               :silver, :lightsalmon, :lightseagreen, :teal, :royalblue1, :darkorchid4]

markercycle = [:rect, :utriangle, :star5, :circle, :cross, :+, :pentagon, :ltriangle, :airplane, :diamond, :star4]
markercycle = repeat(markercycle, inner=2)

function make_axes(fig, row=1, label=nothing)
    ax_b = Axis(fig[row, 1], xlabel = "Buoyancy \n[cm s⁻²]", ylabel = "z [m]")
    ax_u = Axis(fig[row, 2], xlabel = "x-velocity \n[cm s⁻¹]")
    ax_v = Axis(fig[row, 3], xlabel = "y-velocity \n[cm s⁻¹]")
    ax_e = Axis(fig[row, 4], xlabel = "Turbulent kinetic energy \n[cm² s⁻²]")
    if !isnothing(label)
        ax_t = Axis(fig[row, 5])
        xlims!(0, 1)
        ylims!(0, 1)
        hidespines!(ax_t)
        hidedecorations!(ax_t)
        text!(ax_t, label, justification=:left, align=(:left, :center), position=(0, 0.5))
    end
    return (ax_b, ax_u, ax_v, ax_e)
end

function plot_fields!(axs, label, color, b, e, u=zeros(size(b)), v=zeros(size(b)); linewidth=2, linestyle=:solid)
    grid = first(values(observation_library)).grid
    z = znodes(Center, grid)
    b, u, v, e = Tuple(Array(f) for f in (b, u, v, e))

    for (q, name) in zip((b, u, v, e), ("b", "u", "v", "e"))
        any(isnan.(q)) && @warn("NaNs found in $label $(name)!")
    end

    ## Note unit conversions below, eg m s⁻¹ -> cm s⁻¹:cyan
    lines!(axs[1], 1e2 * b, z; color, linestyle, label, linewidth) 
    lines!(axs[2], 1e2 * u, z; color, linestyle, label, linewidth)
    lines!(axs[3], 1e2 * v, z; color, linestyle, label, linewidth)
    lines!(axs[4], 1e4 * e, z; color, linestyle, label, linewidth)

    return nothing
end

function compare_model_observations(model_label="modeled")
    fig = Figure(resolution=(1200, 1200))
    for (c, case) in enumerate(cases)
        label = replace(case, "_" => "\n")
        axs = make_axes(fig, c, label)
        observed = observed_data[c]
        mean_modeled = mean_modeled_data[c]
        best_modeled = best_modeled_data[c]
        latest_best_modeled = latest_best_modeled_data[c]
        worst_modeled = worst_modeled_data[c]

        plot_fields!(axs, "observed at t = " * prettytime(times[end]), (:gray23, 0.6), observed...; linewidth=4)
        plot_fields!(axs, "mean",        :navy,        mean_modeled...)
        plot_fields!(axs, "latest best", :aquamarine4, latest_best_modeled...)
        plot_fields!(axs, "worst",       :orangered3,  worst_modeled...)

        # plot_fields!(axs, "best",        :purple1,       best_modeled...)

        fig[1, 6] = Legend(fig, axs[1]) 
    end

    return fig
end

function min_max_parameters(summary)
    names = keys(summary.ensemble_mean)
    Nens = length(summary.parameters)
    parameter_matrix = [summary.parameters[k][name] for name in names, k = 1:Nens]
    θ_min = minimum(parameter_matrix, dims=2)
    θ_max = maximum(parameter_matrix, dims=2)
    return θ_min, θ_max
end

function visualize_parameter_evolution(eki)
    summaries = eki.iteration_summaries
    Niters = length(summaries)
    names = eki.inverse_problem.free_parameters.names
    θ_mean = NamedTuple(name => map(s -> s.ensemble_mean[name], summaries) for name in names)

    k_best(s) = findmin(s.mean_square_errors)[2]
    θ_best = NamedTuple(name => map(s -> s.parameters[k_best(s)][name], summaries) for name in names)

    θ_min_max = [min_max_parameters(s) for s in summaries]
    θ_min = [[θn[1][i] for θn in θ_min_max] for i in 1:length(names)]
    θ_max = [[θn[2][i] for θn in θ_min_max] for i in 1:length(names)]

    θᵢ = NamedTuple(name => first(θ_mean[name]) for name in names)
    Δθ = NamedTuple(name => (θ_mean[name] .- θᵢ[name]) ./ θᵢ[name] for name in names)
    iterations = 0:length(summaries)-1

    fig = Figure(resolution=(1200, 1200))
    ax1 = Axis(fig[1:3, 1], xlabel = "Iteration", ylabel = "Δθ")
    for (i, name) in enumerate(names)
        label = string(name)
        marker = markercycle[i]
        color = colorcycle[i]
        scatterlines!(ax1, iterations, parent(Δθ[name]); marker, color=(color, 0.8), label, linewidth=4)
    end

    fig[1:3, 2] = Legend(fig, ax1)

    Nparts = 3
    Nθpart = floor(Int, length(names) / Nparts)

    for p in 1:Nparts
        axp = Axis(fig[p+3, 1], xlabel = "Iteration", ylabel = "θ")

        if p == Nparts
            np = UnitRange((p-1) * Nθpart + 1, length(names))
        else
            np = UnitRange((p-1) * Nθpart + 1, p * Nθpart)
        end

        partnames = names[np]

        for (n, name) in enumerate(partnames)
            i = (p - 1) * Nθpart + n
            label = string(name)
            marker = markercycle[i]
            color = colorcycle[i]
            scatterlines!(axp, iterations, parent(θ_mean[name]); marker, color=(color, 0.6), label, linewidth=4)
            lines!(axp, iterations, parent(θ_best[name]); color, linewidth=2)
            band!(axp, iterations, parent(θ_min[i]), parent(θ_max[i]), color=(color, 0.3))
        end

        fig[p+3, 2] = Legend(fig, axp)
    end

    display(fig)

    return nothing
end

latest_summary = eki.iteration_summaries[end]
best_error, k_min = findmin(latest_summary.mean_square_errors)
best_parameters = latest_summary.parameters[k_min]

function latest_best_run!(eki)
    latest_summary = eki.iteration_summaries[end]
    min_error, k_min = findmin(latest_summary.mean_square_errors)
    max_error, k_max = findmax(latest_summary.mean_square_errors)

    if min_error < best_error
        global best_error
        global best_parameters
        best_parameters = latest_summary.parameters[k_min]
        best_error = min_error
    else
        @warn "Parameters did not improve over iteration $(eki.iteration)."
    end

    @show latest_summary

    θ = [latest_summary.ensemble_mean,
         best_parameters,
         latest_summary.parameters[k_min],
         latest_summary.parameters[k_max]]

    @info "Executing a forward run for plotting purposes..."
    forward_run!(eki.inverse_problem, θ)
    @info "  ... done with the forward run."

    i = eki.iteration
    fig = compare_model_observations("modeled after $i iterations")
    display(fig)

    return nothing
end

#####
##### Visualize observations
#####

fig = Figure(resolution=(1200, 1200))
linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
all_axs = []

for (o, observation) in enumerate(observations)
    axs_label = replace(cases[o], "_" => "\n")
    axs = make_axes(fig, o, axs_label)
    append!(all_axs, axs)
    for (n, t) in enumerate(times)
        linestyle = linestyles[o]
        label = "t = " * prettytime(t)
        names = keys(observation.field_time_serieses)
        data = map(name -> interior(observation.field_time_serieses[name][n])[1, 1, :], names)
        plot_fields!(axs, label, colorcycle[n], data...)
    end
end

display(fig)

#####
##### Calibrate
#####

# Initial state after 0 iterations
latest_best_run!(eki)

# Continuously update
for i = 1:100
    @info "Performing two iterations..."
    start_time = time_ns()
    iterate!(eki, iterations=2)
    elapsed = 1e-9 * (time_ns() - start_time)
    @info string("   done. (", prettytime(elapsed), ")")
    latest_best_run!(eki)
    visualize_parameter_evolution(eki)
end

