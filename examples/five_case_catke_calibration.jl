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

#case_path(case) = @datadep_str("two_day_suite_1m/$(case)_instantaneous_statistics.jld2")
case_path(case) = @datadep_str("four_day_suite_1m/$(case)_instantaneous_statistics.jld2")

times = [6hours, 24hours, 96hours]
field_names = (:b, :e, :u, :v)
regrid_size = nothing #(1, 1, 32)

transformation = (b = ZScore(),
                 u = ZScore(),
                 v = ZScore(),
                 e = RescaledZScore(1e-1))

observation_library = Dict()

# Don't optimize u, v for free_convection
observation_library["free_convection"] =
    SyntheticObservations(case_path("free_convection"); normalization, times, regrid_size,
                          field_names = (:b, :e))
                                                                
# Don't optimize v for non-rotating cases
observation_library["strong_wind_no_rotation"] =
    SyntheticObservations(case_path("strong_wind_no_rotation"); normalization, times, regrid_size,
                          field_names = (:b, :e, :u))

# The rest are standard
for case in ["strong_wind", "strong_wind_weak_cooling", "weak_wind_strong_cooling"]
    observation_library[case] = SyntheticObservations(case_path(case); field_names, normalization, times, regrid_size)
end

cases = [
         #"free_convection",
         #"weak_wind_strong_cooling",
         #"strong_wind_weak_cooling",
         "strong_wind",
         "strong_wind_no_rotation",
        ]

observations = [observation_library[case] for case in cases]
     
#####
##### Simulation
#####

mixing_length = MixingLength(Cᴬu   = 0.0,
                             Cᴬc   = 0.0,
                             Cᴬe   = 0.0,
                             Cᴸᵇ   = 1.36,
                             Cᴷu⁻  = 0.101,
                             Cᴷc⁻  = 0.0574,
                             Cᴷe⁻  = 3.32,
                             Cᵟu   = 0.296,
                             Cᵟc   = 1.32,
                             Cᵟe   = 1.49,
                             CᴷRiᶜ = 2.0,
                             Cᴷuʳ  = 0.0,
                             Cᴷcʳ  = 0.0,
                             Cᴷeʳ  = 0.0)

surface_TKE_flux = SurfaceTKEFlux(CᵂwΔ=4.74, Cᵂu★=2.76)
                                  
catke = CATKEVerticalDiffusivity(; Cᴰ=1.779, mixing_length)

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 1000,
                                              architecture = GPU(),
                                              tracers = (:b, :e),
                                              closure = catke)

# `ensemble_column_model_simulation` sets up `simulation`
# with a `FluxBoundaryCondition` array initialized to 0 and a default
# time-step. We modify these for our particular problem,

simulation.Δt = 20.0

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

#####
##### Calibration
#####

mass = 0.5
prior_library = Dict()
prior_library[:Cᴰ]    = ScaledLogitNormal(; bounds=(0, 5), interval=(1.6, 1.9), mass)
prior_library[:CᵂwΔ]  = ScaledLogitNormal(; bounds=(0, 9), interval=(4.6, 4.8), mass)
prior_library[:Cᵂu★]  = ScaledLogitNormal(; bounds=(0, 9), interval=(2.6, 2.9), mass)
prior_library[:Cᴸᵇ]   = ScaledLogitNormal(; bounds=(0, 5), interval=(1.0, 1.6), mass)

prior_library[:Cᴷu⁻]  = ScaledLogitNormal(; bounds=(0.05, 0.2))
prior_library[:Cᴷc⁻]  = ScaledLogitNormal(; bounds=(0.05, 0.1))
prior_library[:Cᴷe⁻]  = ScaledLogitNormal(; bounds=(3, 4))

#prior_library[:Cᴷu⁻]  = ScaledLogitNormal(; bounds=(0, 1), interval=(0.05, 0.2), mass)
#prior_library[:Cᴷc⁻]  = ScaledLogitNormal(; bounds=(0, 1), interval=(0.01, 0.2), mass)
#prior_library[:Cᴷe⁻]  = ScaledLogitNormal(; bounds=(0, 9), interval=(3, 4), mass)

prior_library[:Cᵟu]  = ScaledLogitNormal(; bounds=(0, 1), interval=(0.1, 0.5), mass)
prior_library[:Cᵟc]  = ScaledLogitNormal(; bounds=(0, 3), interval=(1.0, 1.5), mass)
prior_library[:Cᵟe]  = ScaledLogitNormal(; bounds=(0, 3), interval=(1.0, 2.0), mass)

prior_library[:Cᴷuʳ]  = ScaledLogitNormal(; bounds=(-1, 1)) #, interval=(-0.05, 0.05), mass)
prior_library[:Cᴷcʳ]  = ScaledLogitNormal(; bounds=(-1, 1)) #, interval=(-0.05, 0.05), mass)
prior_library[:Cᴷeʳ]  = ScaledLogitNormal(; bounds=(-1, 1)) #, interval=(-0.05, 0.05), mass)

prior_library[:CᴷRiʷ] = ScaledLogitNormal(; bounds=(0, 2.0))
prior_library[:CᴷRiᶜ] = ScaledLogitNormal(; bounds=(2, 3))

prior_library[:Cᴬu]   = ScaledLogitNormal(bounds=(0, 0.1))
prior_library[:Cᴬc]   = ScaledLogitNormal(bounds=(0, 10))
prior_library[:Cᴬe]   = ScaledLogitNormal(bounds=(0, 0.1))

# No convective adjustment:
constant_Ri_parameters = (:Cᴰ, :CᵂwΔ, :Cᵂu★, :Cᴸᵇ, :Cᴷu⁻, :Cᴷc⁻, :Cᴷe⁻, :Cᵟu, :Cᵟc, :Cᵟe)
variable_Ri_parameters = (:Cᴷuʳ, :Cᴷcʳ, :Cᴷeʳ, :CᴷRiʷ, :CᴷRiᶜ, :Cᴰ, :Cᴸᵇ, :CᵂwΔ, :Cᵂu★)
convective_adjustment_parameters = (:Cᴬu, :Cᴬc, :Cᴬe)

free_parameters = FreeParameters(prior_library, names=tuple(:Cᴷu⁻, :Cᴷc⁻, :Cᴷe⁻, variable_Ri_parameters...))
#free_parameters = FreeParameters(prior_library, names=variable_Ri_parameters)
calibration = InverseProblem(observations, simulation, free_parameters)

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 1e-1,
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

mean_modeled_data        = [NamedTuple(n => get_modeled_case(c, n, 1) for n in field_names) for c = 1:length(observations)]
best_modeled_data        = [NamedTuple(n => get_modeled_case(c, n, 2) for n in field_names) for c = 1:length(observations)]
latest_best_modeled_data = [NamedTuple(n => get_modeled_case(c, n, 3) for n in field_names) for c = 1:length(observations)]
worst_modeled_data       = [NamedTuple(n => get_modeled_case(c, n, 4) for n in field_names) for c = 1:length(observations)]

colorcycle =  [:black, :darkblue, :orange, :pink1, :seagreen, :magenta2, :red4, :khaki1,   :darkgreen, :bisque4,
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
        plot_fields!(axs, "ensemble model mean",                       :navy,          mean_modeled...)
        plot_fields!(axs, "ensemble model best",                       :purple1,       best_modeled...)
        plot_fields!(axs, "ensemble model latest best",                :aquamarine4,   latest_best_modeled...)
        plot_fields!(axs, "ensemble model worst",                      :orangered3,    worst_modeled...)
        [axislegend(ax, position=:rb, labelsize=10) for ax in axs]
    end

    return fig
end

function visualize_parameter_evolution(eki)
    summaries = eki.iteration_summaries
    parameter_names = keys(first(summaries).ensemble_mean)
    ensemble_means = NamedTuple(name => map(summary -> summary.ensemble_mean[name], summaries)
                                for name in parameter_names)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = "Iteration", ylabel = "Parameter value")

    for (i, name) in enumerate(parameter_names)
        label = string(name)
        marker = markercycle[i]
        color = colorcycle[i]
        scatterlines!(ax, 0:length(summaries)-1, parent(ensemble_means[name]); marker, color, label)
    end

    axislegend(ax, position=:rb)
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
    @show min_error k_min max_error k_max

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
    @info "Iterating..."
    start_time = time_ns()
    iterate!(eki)
    elapsed = 1e-9 * (time_ns() - start_time)
    @info string("   done. (", prettytime(elapsed), ")")
    latest_best_run!(eki)
    visualize_parameter_evolution(eki)
end

