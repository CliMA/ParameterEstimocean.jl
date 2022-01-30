# # Perfect CAKTE calibration with Ensemble Kalman Inversion

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units
using OceanTurbulenceParameterEstimation
using LinearAlgebra, CairoMakie, DataDeps, Distributions

using ElectronDisplay

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    MixingLength

#####
##### Compile LESbrary
#####

case_path(case) = @datadep_str("two_day_suite_2m/$(case)_instantaneous_statistics.jld2")

times = [2hours, 12hours]
field_names = (:b, :e, :u, :v)

normalization = (b = ZScore(),
                 u = RescaledZScore(0.1),
                 v = RescaledZScore(0.1),
                 e = RescaledZScore(1e-3))

observation_library = Dict()

# Don't optimize u, v for free_convection
observation_library["free_convection"] =
    SyntheticObservations(case_path("free_convection"); normalization, times, field_names = (:b, :e))
                                                                
# Don't optimize v for non-rotating cases
observation_library["strong_wind_no_rotation"] =
    SyntheticObservations(case_path("strong_wind_no_rotation"); normalization, times, field_names = (:b, :e, :u))

# The rest are standard
for case in ["strong_wind", "strong_wind_weak_cooling", "weak_wind_strong_cooling"]
    observation_library[case] = SyntheticObservations(case_path(case); field_names, normalization, times)
end

#=
cases = ["free_convection",
         "strong_wind",
         "strong_wind_no_rotation",
         "strong_wind_weak_cooling",
         "weak_wind_strong_cooling"]
=#

cases = ["free_convection",
         "weak_wind_strong_cooling"]

observations = [observation_library[case] for case in cases]
     
#####
##### Simulation
#####

catke_mixing_length = MixingLength(Cᴬu=0.0, Cᴬc=0.0, Cᴬe=0.0,
                                   Cᴷuʳ=0.0, Cᴷcʳ=0.0, Cᴷeʳ=0.0)
                                  
catke = CATKEVerticalDiffusivity(mixing_length=catke_mixing_length)

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 100,
                                              architecture = GPU(),
                                              tracers = (:b, :e),
                                              closure = catke)

# `ensemble_column_model_simulation` sets up `simulation`
# with a `FluxBoundaryCondition` array initialized to 0 and a default
# time-step. We modify these for our particular problem,

simulation.Δt = 1.0

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

prior_library = Dict()
prior_library[:Cᴰ]    = lognormal_with_mean_std(2.9,  0.1)
prior_library[:CᵂwΔ]  = lognormal_with_mean_std(3.5,  0.1)
prior_library[:Cᵂu★]  = lognormal_with_mean_std(1.0,  0.1)
prior_library[:Cᴸᵇ]   = lognormal_with_mean_std(1.0, 0.2)
prior_library[:Cᴬu]   = ConstrainedNormal(1e-3, 0.01, 0.0, 1.0)
prior_library[:Cᴬc]   = ConstrainedNormal(1e-3, 0.01, 0.0, 1.0)
prior_library[:Cᴬe]   = ConstrainedNormal(1e-3, 0.01, 0.0, 1.0)
prior_library[:Cᴷu⁻]  = ConstrainedNormal(0.1, 0.1, 0.0, 2.0)
prior_library[:Cᴷc⁻]  = ConstrainedNormal(0.4, 0.1, 0.0, 2.0)
prior_library[:Cᴷe⁻]  = ConstrainedNormal(0.2, 0.1, 0.0, 2.0)
prior_library[:Cᴷuʳ]  = Normal(0.01, 0.01)
prior_library[:Cᴷcʳ]  = Normal(0.01, 0.01)
prior_library[:Cᴷeʳ]  = Normal(0.01, 0.01)
prior_library[:CᴷRiʷ] = lognormal_with_mean_std(0.1, 0.05)
prior_library[:CᴷRiᶜ] = Normal(0.2, 0.1)


# No convective adjustment:
constant_Ri_parameters = (:Cᴰ, :CᵂwΔ, :Cᵂu★, :Cᴸᵇ, :Cᴷu⁻, :Cᴷc⁻, :Cᴷe⁻)
variable_Ri_parameters = tuple(constant_Ri_parameters..., :Cᴷuʳ, :Cᴷcʳ, :Cᴷeʳ, :CᴷRiʷ, :CᴷRiᶜ)
constant_Ri_convective_adjustment_parameters = tuple(constant_Ri_parameters..., :Cᴬu, :Cᴬc, :Cᴬe)
variable_Ri_convective_adjustment_parameters = keys(prior_library)

free_parameters = FreeParameters(prior_library, names=constant_Ri_parameters)
calibration = InverseProblem(observations, simulation, free_parameters)

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 1e-2,
                              resampler = NaNResampler(abort_fraction=0.8))

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

function get_modeled_case(icase, name)
    model_time_serieses = calibration.time_series_collector.field_time_serieses 
    field = getproperty(model_time_serieses, name)[Nt]
    return Array(interior(field))[1, icase, :]
end

modeled_data = [NamedTuple(n => get_modeled_case(c, n) for n in field_names) for c = 1:length(observations)]

#              1,      2,          3,         4,       5,      6,         7,         8,          9,         10          11
colorcycle =  [:black, :red,       :darkblue, :orange, :pink1, :seagreen, :magenta2, :red4,      :khaki1,   :darkgreen, :bisque4]
markercycle = [:rect,  :utriangle, :star5,    :circle, :cross, :+,        :pentagon, :ltriangle, :airplane, :diamond,   :star4]

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

function plot_fields!(axs, label, color, linestyle, b, e, u=zeros(size(b)), v=zeros(size(b)))
    grid = first(values(observation_library)).grid
    z = znodes(Center, grid)
    ## Note unit conversions below, eg m s⁻¹ -> cm s⁻¹:
    lines!(axs[1], 1e2 * b, z; color, linestyle, label) 
    lines!(axs[2], 1e2 * u, z; color, linestyle, label)
    lines!(axs[3], 1e2 * v, z; color, linestyle, label)
    lines!(axs[4], 1e4 * e, z; color, linestyle, label)
    return nothing
end

function compare_model_observations(model_label="modeled")
    fig = Figure(resolution=(1200, 1200))
    for (c, case) in enumerate(cases)
        label = replace(case, "_" => "\n")
        axs = make_axes(fig, c, label)
        observed = observed_data[c]
        modeled = modeled_data[c]

        plot_fields!(axs, "observed at t = " * prettytime(times[end]), :black, :solid, observed...)
        plot_fields!(axs, model_label, :blue, :solid, modeled...)
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

function latest_best_run!(eki)
    latest_summary = eki.iteration_summaries[end]

    # @show best_parameters = latest_summary.ensemble_mean
    # @show extrema(latest_summary.mean_square_errors)
    # @show mean(latest_summary.mean_square_errors)
    
    latest_best_parameters = latest_summary.ensemble_mean
    forward_run!(eki.inverse_problem, latest_best_parameters)
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
        plot_fields!(axs, label, colorcycle[n], :solid, data...)
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

