# # CAKTE calibration with Ensemble Kalman Inversion using LESbrary data 

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add ParameterEstimocean, Oceananigans, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units
using ParameterEstimocean
using LinearAlgebra, CairoMakie, DataDeps, Distributions

using Oceananigans.TurbulenceClosures: RiBasedVerticalDiffusivity

# # Using LESbrary data
#
# `ParameterEstimocean.jl` provides paths to synthetic observations
# derived from high-fidelity large eddy simulations. In this example, we illustrate
# calibration of a turbulence parameterization to one of these simulations:

datapath = datadep"two_day_suite_1m/strong_wind_instantaneous_statistics.jld2"
times = [2hours, 12hours, 24hours]
field_names = (:b, :u, :v)
transformation = ZScore()
regrid = (1, 1, 32)

observations = SyntheticObservations(datapath; field_names, times, transformation, regrid)

# Let's take a look at the observations. We define a few
# plotting utilities along the way to use later in the example:

colorcycle = [:black, :red, :darkblue, :orange, :pink1, :seagreen, :magenta2]
markercycle = [:rect, :utriangle, :star5, :circle, :cross, :+, :pentagon]

function make_figure_axes()
    fig = Figure(resolution=(1200, 400))
    ax_b = Axis(fig[1, 1], xlabel = "Buoyancy \n[cm s⁻²]", ylabel = "z [m]")
    ax_u = Axis(fig[1, 2], xlabel = "x-velocity, u \n[cm s⁻¹]")
    ax_v = Axis(fig[1, 3], xlabel = "y-velocity, v \n[cm s⁻¹]")
    return fig, (ax_b, ax_u, ax_v)
end

function plot_fields!(axs, b, u, v, label, color)
    z = znodes(Center, b.grid)
    ## Note unit conversions below, eg m s⁻² -> cm s⁻²:
    lines!(axs[1], 1e2 * interior(b, 1, 1, :), z; color, label)
    lines!(axs[2], 1e2 * interior(u, 1, 1, :), z; color, label)
    lines!(axs[3], 1e2 * interior(v, 1, 1, :), z; color, label)
    return nothing
end

# And then plot the evolution of the observed fields,

fig, axs = make_figure_axes()

for (i, t) in enumerate(times)
    fields = map(name -> observations.field_time_serieses[name][i], field_names)
    plot_fields!(axs, fields..., "t = " * prettytime(t), colorcycle[i])
end

[axislegend(ax, position=:rb, merge=true, fontsize=10) for ax in axs]

save("lesbrary_synthetic_observations.svg", fig); nothing # hide

# ![](lesbrary_synthetic_observations.svg)

# Behold, boundary layer turbulence!
# 
# # Calibration
#
# Next, we build a simulation of an ensemble of column models to calibrate
# the closure using Ensemble Kalman Inversion. We configure the closure without convective
# adjustment and with constant (rather than Richardson-number-dependent)
# diffusivity parameters.

ri_based_closure = RiBasedVerticalDiffusivity()

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 60,
                                              architecture = CPU(),
                                              tracers = (:b, :e),
                                              closure = ri_based_closure)

# The simulation is initialized with neutral boundary conditions
# and a default time-step, which we modify for our particular problem:

Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

simulation.Δt = 20minutes

Qᵘ .= observations.metadata.parameters.momentum_flux
Qᵇ .= observations.metadata.parameters.buoyancy_flux
N² .= observations.metadata.parameters.N²_deep

# We identify a subset of the CATKE parameters to calibrate by specifying
# parameter names and prior distributions:

priors = (ν₀   = lognormal(mean=0.01, std=0.005),
          κ₀   = lognormal(mean=0.1,  std=0.05),
          Ri₀ν = Normal(-0.5, 1.0),
          Ri₀κ = Normal(-0.5, 1.0),
          Riᵟν = lognormal(mean=1.0,  std=0.5),
          Riᵟκ = lognormal(mean=1.0,  std=0.5))

free_parameters = FreeParameters(priors)

# TODO: explain the meaning of each parameter
# The prior information comes from experience, prior calibration runs,
# and educated guesses.

calibration = InverseProblem(observations, simulation, free_parameters)

# Next, we calibrate, using a relatively large noise to reflect our
# uncertainty about how close the observations and model can really get,

eki = EnsembleKalmanInversion(calibration; convergence_rate=0.8)
iterate!(eki; iterations = 10)

@show eki.iteration_summaries[end]

# # Results
#
# To analyze the reuslts, we build a new simulation with just one ensemble member
# to evaluate pasome utilities for analyzing the results:

Nt = length(observations.times)
Niter = length(eki.iteration_summaries) - 1
modeled_time_serieses = calibration.time_series_collector.field_time_serieses 
observed = map(name -> observations.field_time_serieses[name][Nt], field_names)
modeled = map(name -> modeled_time_serieses[name][Nt], field_names)

function compare_model_observations(model_label="modeled")
    fig, axs = make_figure_axes()
    plot_fields!(axs, observed..., "observed at t = " * prettytime(times[end]), :black)
    plot_fields!(axs, modeled..., model_label, :blue)
    [axislegend(ax, position=:rb, merge=true, fontsize=10) for ax in axs]
    return fig
end

# Now we execute forward runs for the initial ensemble mean,

initial_parameters = eki.iteration_summaries[0].ensemble_mean
forward_run!(calibration, initial_parameters)
fig = compare_model_observations("modeled after 0 iterations")

save("model_observation_comparison_iteration_0.svg", fig); nothing # hide

# ![](model_observation_comparison_iteration_0.svg)

# and the final ensemble mean, representing our "best" parameter set,

best_parameters = eki.iteration_summaries[end].ensemble_mean
forward_run!(calibration, best_parameters)
fig = compare_model_observations("modeled after $Niter iterations")

save("model_observation_comparison_final_iteration.svg", fig); nothing # hide

# ![](model_observation_comparison_final_iteration.svg)

# ## Parameter evolution
#
# To understand how results changed over the EKI iterations,
# we look at the evoluation of the ensemble means,

ensemble_means = NamedTuple(n => map(summary -> summary.ensemble_mean[n], eki.iteration_summaries)
                            for n in calibration.free_parameters.names)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Ensemble Kalman iteration", ylabel = "Parameter value")

for (i, name) in enumerate(calibration.free_parameters.names)
    label = string(name)
    marker = markercycle[i]
    color = colorcycle[i]
    scatterlines!(ax, 0:Niter, parent(ensemble_means[name]); marker, color, label)
end

axislegend(ax, position=:rb)

save("lesbrary_catke_parameter_evolution.svg", fig); nothing # hide

# ![](lesbrary_catke_parameter_evolution.svg)
