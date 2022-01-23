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
using ElectronDisplay

using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity, MixingLength

# # Using LESbrary data
#
# `OceanTurbulenceParameterEstimation.jl` provides paths to synthetic observations
# derived from high-fidelity large eddy simulations. In this example, we illustrate
# calibration of a turbulence parameterization to one of these:

data_path = datadep"two_day_suite_4m/strong_wind_instantaneous_statistics.jld2"

times = [2hours, 12hours, 24hours, 36hours]

observations = SyntheticObservations(data_path;
                                     field_names = (:u, :v, :b, :e),
                                     normalize = ZScore,
                                     times)

fig = Figure()

ax_b = Axis(fig[1, 1], xlabel = "Buoyancy\n[10⁻⁴ m s⁻²]", ylabel = "z [m]")
ax_u = Axis(fig[1, 2], xlabel = "Velocities\n[cm s⁻¹]")
ax_e = Axis(fig[1, 3], xlabel = "Turbulent kinetic energy\n[10⁻⁴ m² s⁻²]")

z = znodes(Center, observations.grid)

colorcycle = [:black, :red, :blue, :orange, :pink]

for i = 1:length(observations.times)
    b = observations.field_time_serieses.b[i]
    e = observations.field_time_serieses.e[i]
    u = observations.field_time_serieses.u[i]
    v = observations.field_time_serieses.v[i]
    t = observations.times[i]

    label = "t = " * prettytime(t)
    u_label = i == 1 ? "u, " * label : label
    v_label = i == 1 ? "v, " * label : label

    lines!(ax_b, 1e4 * interior(b)[1, 1, :], z; label, color=colorcycle[i]) # convert units m s⁻² -> 10⁻⁴ m s⁻²
    lines!(ax_u, 1e2 * interior(u)[1, 1, :], z; linestyle=:solid, color=colorcycle[i], label=u_label) # convert units m s⁻¹ -> cm s⁻¹
    lines!(ax_u, 1e2 * interior(v)[1, 1, :], z; linestyle=:dash, color=colorcycle[i], label=v_label) # convert units m s⁻¹ -> cm s⁻¹
    lines!(ax_e, 1e4 * interior(e)[1, 1, :], z; label, color=colorcycle[i]) # convert units m² s⁻² -> 10⁻⁴ m² s⁻²
end

axislegend(ax_b, position=:rb)
axislegend(ax_u, position=:lb, merge=true)
axislegend(ax_e, position=:rb)

display(fig)

save("lesbrary_synthetic_observations.svg", fig); nothing # hide

# ![](lesbrary_synthetic_observations.svg)

# Well, that looks like a boundary layer, in some respects.
# 
# # Calibration
#
# Next, we build a simulation of an ensemble of column models to calibrate
# CATKE using Ensemble Kalman Inversion.

catke_mixing_length = MixingLength(Cᴷcʳ=0.0, Cᴷuʳ=0.0, Cᴷeʳ=0.0)
catke = CATKEVerticalDiffusivity(mixing_length=catke_mixing_length)

simulation = ensemble_column_model_simulation(observations;
                                              Nensemble = 10,
                                              architecture = CPU(),
                                              tracers = (:b, :e),
                                              closure = catke)

Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

simulation.Δt = 10.0

Qᵘ .= observations.metadata.parameters.momentum_flux
Qᵇ .= observations.metadata.parameters.buoyancy_flux
N² .= observations.metadata.parameters.N²_deep

# We choose to calibrate a subset of the CATKE parameters,

priors = (Cᴰ = lognormal_with_mean_std(0.05, 0.1),
          Cᵂu★ = lognormal_with_mean_std(0.05, 0.1),
          CᵂwΔ = lognormal_with_mean_std(0.05, 0.1),
          Cᴸᵇ = lognormal_with_mean_std(0.05, 0.1),
          Cᴷu⁻ = lognormal_with_mean_std(0.05, 0.1),
          Cᴷc⁻ = lognormal_with_mean_std(0.05, 0.1),
          Cᴷe⁻ = lognormal_with_mean_std(0.05, 0.1))

free_parameters = FreeParameters(priors)

# and construct an  `InverseProblem`,

calibration = InverseProblem(observations, simulation, free_parameters)

unity_parameters = NamedTuple(name => 1.0 for name in keys(priors))
G = forward_map(calibration, unity_parameters)

@show size(G)

#=
# # Ensemble Kalman Inversion
#
# Next, we construct an `EnsembleKalmanInversion` (EKI) object,

eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-1)

# and perform few iterations to see if we can converge to the true parameter values.

iterate!(eki; iterations = 1)

# Last, we visualize the outputs of EKI calibration.

## Convert everything to a vector
optimal_θ = collect(values(θ★))
ensemble_mean_θ = map(summary -> collect(values(summary.ensemble_mean)), eki.iteration_summaries)
θ_variances = map(summary -> collect(values(summary.ensemble_var)), eki.iteration_summaries)

names = keys(θ★)
absolute_error = NamedTuple(name => map(θ -> θ[p] - θ★[p], ensemble_mean_θ) for (p, name) in enumerate(names))
relative_error = NamedTuple(name => abs.(absolute_error[name]) ./ θ★[name] for name in names)

output_distances = map(θ -> norm(forward_map(calibration, θ)[:, 1:1] - y), ensemble_mean_θ)

fig = Figure()

ax_error = Axis(fig[1, 1], title = "Parameter distance", xlabel = "Iteration", ylabel = "|⟨θₙ⟩ - θ★| / θ★")

for name in names
    lines!(ax_error, 0:eki.iteration, parent(relative_error[name]), linewidth=2, label=string(name))
end

axislegend(ax_error, position=:rt)

lines(fig[1, 2], 0:eki.iteration, parent(output_distances), color = :blue, linewidth = 2,
      axis = (title = "Output distance", xlabel = "Iteration", ylabel = "|G(⟨θₙ⟩) - y|"))

ax3 = Axis(fig[2, 1:2], title = "Parameter convergence", xlabel = "Iteration",
           ylabel = "Relative change ensemble variance", yscale = log10)

for (p, name) in enumerate(free_parameters.names)
    θp_variances = [θ_variances[iter][p] for iter = 0:eki.iteration]
    lines!(ax3, 0:eki.iteration, parent(θp_variances / θp_variances[1]), label = String(name), linewidth = 2)
end

axislegend(ax3, position = :rt)

##display(fig)

save("perfect_catke_calibration_summary.svg", fig); nothing #hide

# ![](perfect_catke_calibration_summary.svg)

final_mean_θ = eki.iteration_summaries[end].ensemble_mean
forward_run!(calibration, [θ★, final_mean_θ])

time_series_collector = calibration.time_series_collector
times = time_series_collector.times

## Extract last save point and plot each solution component
Nt = length(times)

b = time_series_collector.field_time_serieses.b[Nt]
e = time_series_collector.field_time_serieses.e[Nt]
u = time_series_collector.field_time_serieses.u[Nt]
v = time_series_collector.field_time_serieses.v[Nt]

t = times[Nt]
z = znodes(b)

## The ensemble varies along the first, or `x`-dimension:
b★ = 1e4 * interior(b)[1, 1, :]  # convert units m s⁻² -> 10⁻⁴ m s⁻²
b¹ = 1e4 * interior(b)[2, 1, :]  # convert units m s⁻² -> 10⁻⁴ m s⁻²

e★ = 1e4 * interior(e)[1, 1, :]  # convert units m² s⁻² -> 10⁻⁴ m² s⁻²
e¹ = 1e4 * interior(e)[2, 1, :]  # convert units m² s⁻² -> 10⁻⁴ m² s⁻²

u★ = 1e2 * interior(u)[1, 1, :]  # convert units m s⁻¹ -> cm s⁻¹
u¹ = 1e2 * interior(u)[2, 1, :]  # convert units m s⁻¹ -> cm s⁻¹

v★ = 1e2 * interior(v)[1, 1, :]  # convert units m s⁻¹ -> cm s⁻¹
v¹ = 1e2 * interior(v)[2, 1, :]  # convert units m s⁻¹ -> cm s⁻¹

fig = Figure()

ax = Axis(fig[1, 1], xlabel = "Buoyancy\n[10⁻⁴ m s⁻²]", ylabel = "z [m]")
b★_label = "true b at " * prettytime(t)
b¹_label = "b with ⟨θ⟩"
lines!(ax, b★, z; label=b★_label, linewidth=3)
lines!(ax, b¹, z; label=b¹_label, linewidth=2)
axislegend(ax, position=:lb)

ax = Axis(fig[1, 2], xlabel = "Turbulent kinetic energy\n[10⁻⁴ m² s⁻²]")
e★_label = "true e at " * prettytime(t)
e¹_label = "e with ⟨θ⟩"
lines!(ax, e★, z; label=e★_label, linewidth=3)
lines!(ax, e¹, z; label=e¹_label, linewidth=2)
axislegend(ax, position=:lb)

ax = Axis(fig[1, 3], xlabel = "Velocities\n[cm s⁻¹]")
u★_label = "true u at " * prettytime(t)
u¹_label = "u with ⟨θ⟩"
v★_label = "true v"
v¹_label = "v with ⟨θ⟩"
lines!(ax, u★, z; label=u★_label, linewidth=3)
lines!(ax, u¹, z; label=u¹_label, linewidth=2)
lines!(ax, v★, z; label=v★_label, linestyle=:dash, linewidth=3)
lines!(ax, v¹, z; label=v¹_label, linestyle=:dash, linewidth=2)
axislegend(ax, position=:lb)

save("perfect_catke_calibration_particle_realizations.svg", fig); nothing # hide

# ![](perfect_catke_calibration_particle_realizations.svg)

##display(fig)

# And also we plot the the distributions of the various model ensembles for few EKI iterations to see
# if and how well they converge to the true diffusivity values.

fig = Figure()

ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[2, 1], xlabel = "Cᴬu [m² s⁻¹]", ylabel = "Cᴬc [m² s⁻¹]")
ax3 = Axis(fig[2, 2])
scatters = []
labels = String[]

for iteration in [0, 2, 10, 20]
    ## Make parameter matrix
    parameters = eki.iteration_summaries[iteration].parameters
    Nensemble = length(parameters)
    parameter_ensemble_matrix = [parameters[i][j] for i=1:Nensemble, j=1:2]

    label = iteration == 0 ? "Initial ensemble" : "Iteration $iteration"
    push!(labels, label)
    push!(scatters, scatter!(ax2, parameter_ensemble_matrix))
    density!(ax1, parameter_ensemble_matrix[:, 1])
    density!(ax3, parameter_ensemble_matrix[:, 2], direction = :y)
end

vlines!(ax1, [θ★.Cᴬu], color = :red)
vlines!(ax2, [θ★.Cᴬu], color = :red)
hlines!(ax2, [θ★.Cᴬc], color = :red)
hlines!(ax3, [θ★.Cᴬc], color = :red)

colsize!(fig.layout, 1, Fixed(300))
colsize!(fig.layout, 2, Fixed(200))
rowsize!(fig.layout, 1, Fixed(200))
rowsize!(fig.layout, 2, Fixed(300))

Legend(fig[1, 2], scatters, labels, position = :lb)

hidedecorations!(ax1, grid = false)
hidedecorations!(ax3, grid = false)

xlims!(ax1, 0.025, 0.125)
xlims!(ax2, 0.025, 0.125)
ylims!(ax2, 0.35, 0.9)
ylims!(ax3, 0.35, 0.9)

##display(fig)

save("perfect_catke_calibration_parameter_distributions.svg", fig); nothing # hide

# ![](perfect_catke_calibration_parameter_distributions.svg)

# Hint: if using a REPL or notebook, try
# `using Pkg; Pkg.add("ElectronDisplay"); using ElectronDisplay; display(fig)`
# To see the figure in a window.
=#
