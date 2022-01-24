# # CAKTE calibration with Ensemble Kalman Inversion using LESbrary data 

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units
using OceanTurbulenceParameterEstimation
using LinearAlgebra
using CairoMakie
using DataDeps

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity, MixingLength

# # Using LESbrary data
#
# `OceanTurbulenceParameterEstimation.jl` provides paths to synthetic observations
# derived from high-fidelity large eddy simulations. In this example, we illustrate
# calibration of a turbulence parameterization to one of these simulations:

data_path = datadep"two_day_suite_4m/strong_wind_instantaneous_statistics.jld2"

times = [2hours, 6hours, 18hours]

observations = SyntheticObservations(data_path;
                                     field_names = (:u, :v, :b, :e),
                                     normalize = ZScore,
                                     times)

# Let's take a look at the observations. We define a few
# plotting utilities along the way to use later in the example:

colorcycle = [:black, :red, :darkblue, :orange, :pink1, :seagreen, :magenta2]
markercycle = [:rect, :utriangle, :star5, :circle, :cross, :+, :pentagon]

function make_figure_axes()
    fig = Figure()
    ax_b = Axis(fig[1, 1], xlabel = "Buoyancy\n[10⁻⁴ m s⁻²]", ylabel = "z [m]")
    ax_u = Axis(fig[1, 2], xlabel = "Velocities\n[cm s⁻¹]")
    ax_e = Axis(fig[1, 3], xlabel = "Turbulent kinetic energy\n[10⁻⁴ m² s⁻²]")
    return fig, (ax_b, ax_u, ax_e)
end

function plot_fields!(axs, b, u, v, e, label, u_label, v_label, color)
    z = znodes(Center, observations.grid)
    lines!(axs[1], 1e4 * interior(b)[1, 1, :], z; label, color) # convert units m s⁻² -> 10⁻⁴ m s⁻²
    lines!(axs[2], 1e2 * interior(u)[1, 1, :], z; linestyle=:solid, color, label=u_label) # convert units m s⁻¹ -> cm s⁻¹
    lines!(axs[2], 1e2 * interior(v)[1, 1, :], z; linestyle=:dash, color, label=v_label) # convert units m s⁻¹ -> cm s⁻¹
    lines!(axs[3], 1e4 * interior(e)[1, 1, :], z; label, color) # convert units m² s⁻² -> 10⁻⁴ m² s⁻²
end

fig, axs = make_figure_axes()

for i = 1:length(observations.times)
    b = observations.field_time_serieses.b[i]
    e = observations.field_time_serieses.e[i]
    u = observations.field_time_serieses.u[i]
    v = observations.field_time_serieses.v[i]
    t = observations.times[i]

    label = "t = " * prettytime(t)
    u_label = i == 1 ? "u, " * label : label
    v_label = i == 1 ? "v, " * label : label
    plot_fields!(axs, b, u, v, e, label, u_label, v_label, colorcycle[i])
end

axislegend(axs[1], position=:rb)
axislegend(axs[2], position=:lb, merge=true)
axislegend(axs[3], position=:rb)

save("lesbrary_synthetic_observations.svg", fig); nothing # hide

# ![](lesbrary_synthetic_observations.svg)

# Well, that looks like a boundary layer.
# 
# # Calibration
#
# Next, we build a simulation of an ensemble of column models to calibrate
# CATKE using Ensemble Kalman Inversion.

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

priors = (Cᴰ   = lognormal_with_mean_std(2.5,  0.1),
          Cᵂu★ = lognormal_with_mean_std(1.0,  0.1),
          CᵂwΔ = lognormal_with_mean_std(0.05, 0.1),
          Cᴸᵇ  = lognormal_with_mean_std(0.05, 0.1),
          Cᴷu⁻ = ConstrainedNormal(0.1,  0.05, 0.0, 2.0),
          Cᴷc⁻ = ConstrainedNormal(0.1,  0.05, 0.0, 2.0),
          Cᴷe⁻ = ConstrainedNormal(1.0,  0.1,  0.0, 2.0))

free_parameters = FreeParameters(priors)

calibration = InverseProblem(observations, simulation, free_parameters)

# Next we perform a preliminary calibration by executing one iteration 
# of EnsembleKalmanInversion with a relatively large noise,

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 1e-1,
                              resampler = NaNResampler(abort_fraction=0.1))

iterate!(eki; iterations = 1)

# One iteration won't do much. But let's look at the results anyways
# by executing another run with the ensemble mean parameters:

best_parameters = eki.iteration_summaries[end].ensemble_mean
forward_run!(calibration, best_parameters)

Nt = length(observations.times)
tN = observations.times[Nt]

b_obs = observations.field_time_serieses.b[Nt]
e_obs = observations.field_time_serieses.e[Nt]
u_obs = observations.field_time_serieses.u[Nt]
v_obs = observations.field_time_serieses.v[Nt]

model_time_serieses = calibration.time_series_collector.field_time_serieses 
b_model = model_time_serieses.b[Nt]
e_model = model_time_serieses.e[Nt]
u_model = model_time_serieses.u[Nt]
v_model = model_time_serieses.v[Nt]

function compare_model_observations()
    fig, axs = make_figure_axes()
       
    color = :black
    label = "observed at t = " * prettytime(tN)
    plot_fields!(axs, b_obs, u_obs, v_obs, e_obs, label, "u " * label, "v " * label, color)
    
    color = :blue
    label = "modeled"
    plot_fields!(axs, b_model, u_model, v_model, e_model, label, "u " * label, "v " * label, color)
        
    axislegend(axs[1], position=:rb)
    axislegend(axs[2], position=:lb, merge=true)
    axislegend(axs[3], position=:rb)

    return fig
end

fig = compare_model_observations()

save("model_observation_comparison_iteration_1.svg", fig); nothing # hide

# ![](model_observation_comparison_iteration_1.svg)

# Now let's see if further iterations improve that result...

iterate!(eki; iterations = 10)
best_parameters = eki.iteration_summaries[end].ensemble_mean
forward_run!(calibration, best_parameters)
fig = compare_model_observations()

# Let's see how the parameters evolved:

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

save("model_observation_comparison_iteration_12.svg", fig); nothing # hide

# ![](model_observation_comparison_iteration_12.svg)
