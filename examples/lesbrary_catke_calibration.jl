# # CAKTE calibration with Ensemble Kalman Inversion using LESbrary data 

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units
using OceanTurbulenceParameterEstimation
using LinearAlgebra, CairoMakie, DataDeps

using ElectronDisplay

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity, MixingLength

# # Using LESbrary data
#
# `OceanTurbulenceParameterEstimation.jl` provides paths to synthetic observations
# derived from high-fidelity large eddy simulations. In this example, we illustrate
# calibration of a turbulence parameterization to one of these simulations:

data_path = datadep"two_day_suite_4m/strong_wind_instantaneous_statistics.jld2"
times = [2hours, 6hours, 12hours]
field_names = (:b, :u, :v, :e)

observations = SyntheticObservations(data_path; field_names, times, normalize=ZScore)

# Let's take a look at the observations. We define a few
# plotting utilities along the way to use later in the example:

colorcycle = [:black, :red, :darkblue, :orange, :pink1, :seagreen, :magenta2]
markercycle = [:rect, :utriangle, :star5, :circle, :cross, :+, :pentagon]

function make_figure_axes()
    fig = Figure(resolution=(1200, 400))
    ax_b = Axis(fig[1, 1], xlabel = "Buoyancy \n[cm s⁻²]", ylabel = "z [m]")
    ax_u = Axis(fig[1, 2], xlabel = "x-velocity, u \n[cm s⁻¹]")
    ax_v = Axis(fig[1, 3], xlabel = "y-velocity, v \n[cm s⁻¹]")
    ax_e = Axis(fig[1, 4], xlabel = "Turbulent kinetic energy \n[cm² s⁻²]")
    return fig, (ax_b, ax_u, ax_v, ax_e)
end

function plot_fields!(axs, b, u, v, e, label, color)
    z = znodes(Center, b.grid)
    # Note unit conversions below, eg m s⁻² -> 10⁻⁴ m s⁻²:
    lines!(axs[1], 1e2 * interior(b)[1, 1, :], z; color, label)
    lines!(axs[2], 1e2 * interior(u)[1, 1, :], z; color, label)
    lines!(axs[3], 1e2 * interior(v)[1, 1, :], z; color, label)
    lines!(axs[4], 1e4 * interior(e)[1, 1, :], z; color, label)
    return nothing
end

# And then plot the evolution of the observed fields,

fig, axs = make_figure_axes()

for (i, t) in enumerate(times)
    fields = map(name -> observations.field_time_serieses[name][i], field_names)
    plot_fields!(axs, fields..., "t = " * prettytime(t), colorcycle[i])
end

[axislegend(ax, position=:rb, merge=true, fontsize=10) for ax in axs]

display(fig)
save("lesbrary_synthetic_observations.svg", fig); nothing # hide

# ![](lesbrary_synthetic_observations.svg)

# Behold, boundary layer turbulence!
# 
# # Calibration
#
# Next, we build a simulation of an ensemble of column models to calibrate
# CATKE using Ensemble Kalman Inversion. We configure CATKE without convective
# adjustment and with constant (rather than Richardson-number-dependent)
# diffusivity parameters.

catke_mixing_length = MixingLength(Cᴷcʳ=0.0, Cᴷuʳ=0.0, Cᴷeʳ=0.0)
catke = CATKEVerticalDiffusivity(mixing_length=catke_mixing_length)

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 30,
                                              architecture = CPU(),
                                              tracers = (:b, :e),
                                              closure = catke)

# The simulation is initialized with neutral boundary conditions
# and a default time-step, which we modify for our particular problem:

Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

simulation.Δt = 10.0

Qᵘ .= observations.metadata.parameters.momentum_flux
Qᵇ .= observations.metadata.parameters.buoyancy_flux
N² .= observations.metadata.parameters.N²_deep

# We identify a subset of the CATKE parameters to calibrate by specifying
# parameter names and prior distributions:

priors = (Cᴰ   = lognormal_with_mean_std(2.0, 0.5),
          Cᵂu★ = lognormal_with_mean_std(2.0, 0.5),
          Cᴸᵇ  = lognormal_with_mean_std(0.2, 0.1),
          Cᴷu⁻ = ConstrainedNormal(0.2, 0.1, 0.0, 2.0),
          Cᴷc⁻ = ConstrainedNormal(1.0, 0.1, 0.0, 2.0),
          Cᴷe⁻ = ConstrainedNormal(1.5, 0.3, 0.0, 2.0))

free_parameters = FreeParameters(priors)

# TODO: explain the meaning of each parameter
# The prior information comes from experience, prior calibration runs,
# and educated guesses.

calibration = InverseProblem(observations, simulation, free_parameters)

# Next, we calibrate, using a relatively large noise to reflect our
# uncertainty about how close the observations and model can really get,

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 1e-2,
                              resampler = NaNResampler(abort_fraction=0.5))

iterate!(eki; iterations = 5)

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
display(fig)
save("model_observation_comparison_iteration_0.svg", fig); nothing # hide

# and the final ensemble mean, representing our "best" parameter set,

best_parameters = eki.iteration_summaries[end].ensemble_mean
forward_run!(calibration, best_parameters)
fig = compare_model_observations("modeled after $Niter iterations")
display(fig)
save("model_observation_comparison_iteration_$Niter.svg", fig); nothing # hide

# ## Parameter evolution
#
# To understand how results improved between iteration 0 and iteration 5,
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
display(fig)
save("lesbrary_catke_parameter_evolution.svg", fig); nothing # hide

# ![](lesbrary_catke_parameter_evolution.svg)
