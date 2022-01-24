# # Perfect CAKTE calibration with Ensemble Kalman Inversion

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
# calibration of a turbulence parameterization to one of these:

rotating_data_path = datadep"two_day_suite_4m/strong_wind_instantaneous_statistics.jld2"
non_rotating_data_path = datadep"two_day_suite_4m/strong_wind_no_rotation_instantaneous_statistics.jld2"

times = [2hours, 6hours, 18hours]

field_names = field_names = (:u, :v, :b, :e)
normalize = ZScore
observations = [SyntheticObservations(data_path; field_names, normalize, times)
                for data_path in (rotating_data_path, non_rotating_data_path)]

# Let's take a look at the observations. We define a few
# plotting utilities along the way to use later in the example:

colorcycle = [:black, :red, :darkblue, :orange, :pink1, :seagreen, :magenta2]
markercycle = [:rect, :utriangle, :star5, :circle, :cross, :+, :pentagon]

function make_axes(fig, row=1)
    ax_b = Axis(fig[row, 1], xlabel = "Buoyancy \n[10⁻⁴ m s⁻²]", ylabel = "z [m]")
    ax_u = Axis(fig[row, 2], xlabel = "x-velocity \n[cm s⁻¹]")
    ax_v = Axis(fig[row, 3], xlabel = "y-velocitiy \n[cm s⁻¹]")
    ax_e = Axis(fig[row, 4], xlabel = "Turbulent kinetic energy \n[10⁻⁴ m² s⁻²]")
    return (ax_b, ax_u, ax_v, ax_e)
end

function plot_fields!(axs, b, u, v, e, label, color, linestyle)
    z = znodes(Center, first(observations).grid)

    # Note the unit conversions, eg m s⁻² -> 10⁻⁴ m s⁻² for buoyancy `b`.
    !isnothing(b) && lines!(axs[1], 1e4 * interior(b)[1, 1, :], z; color, linestyle, label) 
    !isnothing(u) && lines!(axs[2], 1e2 * interior(u)[1, 1, :], z; color, linestyle, label="u, " * label)
    !isnothing(v) && lines!(axs[3], 1e2 * interior(v)[1, 1, :], z; color, linestyle, label="v, " * label)
    !isnothing(e) && lines!(axs[4], 1e4 * interior(e)[1, 1, :], z; color, linestyle, label)
end

# And then we make our first plot:

fig = Figure(resolution=(1200, 400))
axs = make_axes(fig)

prefixes, linestyles = ["f=10⁻⁴", "f=0"], [:solid, :dash]

for (o, observation) in enumerate(observations)
    for (n, t) in enumerate(times)
        prefix = prefixes[o]
        linestyle = linestyles[o]
        label = prefix * ", t = " * prettytime(t)

        b = observation.field_time_serieses.b[n]
        e = observation.field_time_serieses.e[n]
        u = observation.field_time_serieses.u[n]
        v = observation.field_time_serieses.v[n]

        plot_fields!(axs, b, u, v, e, label, colorcycle[n], linestyle)
    end
end

[axislegend(ax, position=:rb, labelsize=10) for ax in axs]

save("lesbrary_synthetic_observations.svg", fig); nothing # hide
display(fig)

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

# `ensemble_column_model_simulation` sets up `simulation`
# with a `FluxBoundaryCondition` array initialized to 0 and a default
# time-step. We modify these for our particular problem,

simulation.Δt = 20.0

Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

for (case, obs) in enumerate(observations)
    view(Qᵘ, case, :) .= obs.metadata.parameters.momentum_flux
    view(Qᵇ, case, :) .= obs.metadata.parameters.buoyancy_flux
    view(N², case, :) .= obs.metadata.parameters.N²_deep
end

# ## Free parameters
#
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

# TODO: write out a table of the parameters with their physical meanings...
#
# Finally, we build the `InverseProblem` and `EnsembleKalmanInversion`, and
# execute one iteration of `EnsembleKalmanInversion` with relatively large noise,

calibration = InverseProblem(observations, simulation, free_parameters)

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 1e-1,
                              resampler = NaNResampler(abort_fraction=0.1))

iterate!(eki; iterations = 1)

# One iteration won't do much. But let's look at the results anyways
# by executing another run with the ensemble mean parameters:

best_parameters = eki.iteration_summaries[end].ensemble_mean
forward_run!(calibration, best_parameters)

# We compile the observed fields at the final time
# for both cases,

Nt = length(observations.times)
tN = observations.times[Nt]
field_names = (:b, :u, :v, :e)

observed_data = []
for observation in observations
    time_serieses = observation.field_time_serieses
    case_data = NamedTuple(n => getproperty(time_serieses, name)[Nt] for n in field_names)
    push!(observed_data, case_data)
end

# and same for the model data,

function modeled_case_data(case, name)
    model_time_serieses = calibration.time_series_collector.field_time_serieses 
    field = getproperty(model_time_serieses, name)[Nt]
    return interior(field)[case, :, :]
end

modeled_data = []

for case = 1:2
    case_data = NamedTuple(n => modeled_case_data(case, n) for n in field_names)
    push!(modeled_data, case_data)
end

# Finally we're ready to plot,

function compare_model_observations()
    fig = Figure(resolution=(1200, 800))
    case_axs = [make_axes(fig, row) for row = 1:2]

    for case = 1:2
        axs = case_axes[case]
        obs = observed_data[case]
        modeled = modeled_data[case]

        color = :black
        label = "observed at t = " * prettytime(tN)
        plot_fields!(axs, obs[:b], obs[:u], obs[:v], obs[:e], label, color, :solid)
        
        color = :blue
        label = "modeled"
        plot_fields!(axs, modeled[:b], modeled[:u], modeled[:v], modeled[:e], label, color, :solid)
    end
        
    [axislegend(ax, position=:rb, labelsize=10) for ax in axs]

    return fig
end

fig = compare_model_observations()
display(fig)

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
display(fig)

save("model_observation_comparison_iteration_12.svg", fig); nothing # hide

# ![](model_observation_comparison_iteration_12.svg)
