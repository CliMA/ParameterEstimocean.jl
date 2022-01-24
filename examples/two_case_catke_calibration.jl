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
    z = znodes(Center, b.grid)
    ## Note unit conversions below, eg m s⁻² -> cm s⁻²:
    lines!(axs[1], 1e4 * interior(b)[1, 1, :], z; color, linestyle, label) 
    lines!(axs[2], 1e2 * interior(u)[1, 1, :], z; color, linestyle, label)
    lines!(axs[3], 1e2 * interior(v)[1, 1, :], z; color, linestyle, label)
    lines!(axs[4], 1e4 * interior(e)[1, 1, :], z; color, linestyle, label)
    return nothing
end

# And then we make our first plot:

fig = Figure(resolution=(1200, 400))
axs = make_axes(fig)
prefixes, linestyles = ["f=10⁻⁴", "f=0"], [:solid, :dash]

for (o, observation) in enumerate(observations)
    for (n, t) in enumerate(times)
        prefix = prefixes[o]
        linestyle = linestyles[o]
        label = o == 1 ? prefix * ", t = " * prettytime(t) : prefix
        data = map(time_series -> time_series[n], observation.field_time_serieses)
        plot_fields!(axs, data..., label, colorcycle[n], linestyle)
    end
end

[axislegend(ax, position=:rb, labelsize=10, merge=true) for ax in axs]

save("lesbrary_synthetic_observations.svg", fig); nothing # hide
display(fig)

# ![](lesbrary_synthetic_observations.svg)

# # Calibration
#
# Next, we build a simulation of an ensemble of column models to calibrate
# CATKE using Ensemble Kalman Inversion.

catke_mixing_length = MixingLength()
catke = CATKEVerticalDiffusivity(mixing_length=catke_mixing_length)

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 200,
                                              architecture = CPU(),
                                              tracers = (:b, :e),
                                              closure = catke)

# `ensemble_column_model_simulation` sets up `simulation`
# with a `FluxBoundaryCondition` array initialized to 0 and a default
# time-step. We modify these for our particular problem,

simulation.Δt = 5.0

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

priors = (Cᴰ    = lognormal_with_mean_std(2.5,  0.01),
          Cᵂu★  = lognormal_with_mean_std(1.0,  0.01),
          Cᴸᵇ   = lognormal_with_mean_std(0.05, 0.01),
          Cᴷu⁻  = ConstrainedNormal(0.1,  0.01, 0.0, 2.0),
          Cᴷc⁻  = ConstrainedNormal(0.1,  0.01, 0.0, 2.0),
          Cᴷe⁻  = ConstrainedNormal(1.0,  0.01, 0.0, 2.0),
          Cᴷuʳ  = Normal(1.0, 0.01),
          Cᴷcʳ  = Normal(0.1, 0.01),
          Cᴷeʳ  = Normal(1.0, 0.01),
          CᴷRiʷ = lognormal_with_mean_std(0.05, 0.01),
          CᴷRiᶜ = Normal(0.2, 0.01))

free_parameters = FreeParameters(priors)

# TODO: write out a table of the parameters with their physical meanings...
#
# Finally, we build the `InverseProblem` and `EnsembleKalmanInversion`, and
# execute one iteration of `EnsembleKalmanInversion` with relatively large noise,

calibration = InverseProblem(observations, simulation, free_parameters)

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 1.0,
                              resampler = NaNResampler(abort_fraction=0.5))

# We also build some utilities for comparing observed and model output later on:

Nt = length(times)
observed_data = []
modeled_data = []

for observation in observations
    time_serieses = observation.field_time_serieses
    case_data = NamedTuple(n => getproperty(time_serieses, n)[Nt] for n in field_names)
    push!(observed_data, case_data)
end

function get_modeled_case(case, name)
    model_time_serieses = calibration.time_series_collector.field_time_serieses 
    field = getproperty(model_time_serieses, name)[Nt]
    return interior(field)[case, :, :]
end

for case = 1:2
    push!(modeled_data, NamedTuple(n => get_modeled_case(case, n) for n in field_names))
end

function compare_model_observations(model_label="modeled")
    fig = Figure(resolution=(1200, 800))
    for case = 1:2
        axs = make_axes(fig, case)
        obs = observed_data[case]
        modeled = modeled_data[case]
        plot_fields!(axs, obs..., "observed at t = " * prettytime(times[end]), :black, :solid)
        plot_fields!(axs, modeled..., model_label, :blue, :solid)
        [axislegend(ax, position=:rb, labelsize=10) for ax in axs]
    end
    return fig
end

# Now let's iterate to see if we can find better parameters than our initial guess...

iterate!(eki; iterations = 5)

# To evaluate how 

initial_parameters = eki.iteration_summaries[0].ensemble_mean
best_parameters = eki.iteration_summaries[end].ensemble_mean
Niter = length(eki.iteration_summaries) - 1

forward_run!(calibration, initial_parameters)
fig = compare_model_observations("modeled after 0 iterations")
save("two_case_catke_comparison_iteration_0.svg", fig); nothing # hide
display(fig)

# ![](two_case_catke_comparison_iteration_0.svg", fig)

forward_run!(calibration, best_parameters)
fig = compare_model_observations("modeled after $Niter iterations")
save("two_case_catke_comparison_iteration_$Niter.svg", fig); nothing # hide
display(fig)

# ![](two_case_catke_comparison_iteration_5.svg", fig)

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
