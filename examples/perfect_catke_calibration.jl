# # Perfect CAKTE calibration with Ensemble Kalman Inversion

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, CairoMakie"
# ```

using OceanTurbulenceParameterEstimation, LinearAlgebra, CairoMakie
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity, MixingLength, SurfaceTKEFlux

# using ElectronDisplay

# # Perfect observations of CATKE-driven mixing
#
# Our first task is to generate synthetic observations, using
# a one-dimensional model driven by surface fluxes and with
# turbulent mixing parameterized by CATKE. We use a simplified CATKE
# with no stability function (by setting `Cᴷuʳ = Cᴷcʳ = Cᴷeʳ = 0`)
# and "reasonable", but unrealistic parameters.
# We will only attempt to calibrate a subset of the parameters that we set
# to generate the observations.

## Load utilities
examples_path = joinpath(pathof(OceanTurbulenceParameterEstimation), "..", "..", "examples")
include(joinpath(examples_path, "intro_to_inverse_problems.jl"))

mixing_length = MixingLength(Cᴬu  = 0.1,
                             Cᴬc  = 0.5,
                             Cᴬe  = 0.1,
                             Cᴷu⁻ = 0.1,
                             Cᴷc⁻ = 0.1,
                             Cᴷe⁻ = 0.1,
                             Cᴷuʳ = 0.0,
                             Cᴷcʳ = 0.0,
                             Cᴷeʳ = 0.0)

catke = CATKEVerticalDiffusivity(mixing_length=mixing_length)

## Specify both wind mixing and convection:
data_path = generate_synthetic_observations("catke",
                                            closure = catke,
                                            tracers = (:b, :e),
                                            Nz = 32,
                                            Lz = 64,
                                            Δt = 10.0,
                                            stop_time = 12hours,
                                            overwrite = true,
                                            Qᵘ = -1e-4,
                                            Qᵇ = 1e-8,
                                            N² = 1e-5)

# Next, we load and inspect the observations to make sure they're sensible:

observations = SyntheticObservations(data_path, field_names=(:u, :v, :b, :e), normalize=ZScore)

fig = Figure()

ax_b = Axis(fig[1, 1], xlabel = "Buoyancy\n[10⁻⁴ m s⁻²]", ylabel = "z [m]")
ax_u = Axis(fig[1, 2], xlabel = "Velocities\n[m s⁻¹]", ylabel = "z [m]")
ax_e = Axis(fig[1, 3], xlabel = "Turbulent kinetic energy\n[10⁻⁴ m² s⁻²]", ylabel = "z [m]")

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
    lines!(ax_u, interior(u)[1, 1, :], z; linestyle=:solid, color=colorcycle[i], label=u_label)
    lines!(ax_u, interior(v)[1, 1, :], z; linestyle=:dash, color=colorcycle[i], label=v_label)
    lines!(ax_e, 1e4 * interior(e)[1, 1, :], z; label, color=colorcycle[i]) # convert units m² s⁻² -> 10⁻⁴ m² s⁻²
end

axislegend(ax_b, position=:rb)
axislegend(ax_u, position=:lb, merge=true)
axislegend(ax_e, position=:rb)

##display(fig)

save("synthetic_catke_observations.svg", fig); nothing # hide

# ![](synthetic_catke_observations.svg)

# Well, that looks like a boundary layer, in some respects.
# 
# # Calibration
#
# Next, we build a simulation of an ensemble of column models to calibrate
# CATKE using Ensemble Kalman Inversion.

ensemble_simulation, closure★ = build_ensemble_simulation(observations; Nensemble=100)

# We choose to calibrate a subset of the CATKE parameters,

priors = (Cᴬu = lognormal_with_mean_std(0.05, 0.01),
          Cᴬc = lognormal_with_mean_std(0.05, 0.01),
          Cᴬe = lognormal_with_mean_std(0.05, 0.01))

free_parameters = FreeParameters(priors)

## Perfect parameters...
θ★ = (Cᴬu = catke.mixing_length.Cᴬu,
      Cᴬc = catke.mixing_length.Cᴬc,
      Cᴬe = catke.mixing_length.Cᴬe)

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# y = observation_map(calibration)
# G = forward_map(calibration, θ★)
# @show G[:, 1] ≈ y

# # Ensemble Kalman Inversion
#
# Next, we construct an `EnsembleKalmanInversion` (EKI) object,
#
# The calibration is done here using Ensemble Kalman Inversion. For more information about the 
# algorithm refer to
# [EnsembleKalmanProcesses.jl documentation](
# https://clima.github.io/EnsembleKalmanProcesses.jl/stable/ensemble_kalman_inversion/).

noise_variance = observation_map_variance_across_time(calibration)[1, :, 1] .+ 1e-3
eki = EnsembleKalmanInversion(calibration; noise_covariance = Matrix(Diagonal(noise_variance)))
iterate!(eki; iterations = 20)

# Last, we visualize the outputs of EKI calibration.

## Convert everything to a vector
optimal_θ = collect(values(θ★))
ensemble_mean_θ = map(summary -> collect(values(summary.ensemble_mean)), eki.iteration_summaries)
θ_variances = map(summary -> collect(values(summary.ensemble_var)), eki.iteration_summaries)

names = keys(θ★)
absolute_error = NamedTuple(name => map(θ -> θ[p] - θ★[p], ensemble_mean_θ) for (p, name) in enumerate(names))
relative_error = NamedTuple(name => absolute_error[name] ./ θ★[name] for name in names)

output_distances = map(θ -> norm(forward_map(calibration, θ)[:, 1:1] - y), ensemble_mean_θ)

fig = Figure()

ax_error = Axis(fig[1, 1], title = "Parameter distance", xlabel = "Iteration", ylabel = "|⟨θₙ⟩ - θ★|")

for name in names
    lines!(ax_error, relative_error[name], linewidth=2, label=string(name))
end

axislegend(ax_error, position=:rt)

lines(fig[1, 2], output_distances, color = :blue, linewidth = 2,
      axis = (title = "Output distance", xlabel = "Iteration", ylabel = "|G(⟨θₙ⟩) - y|"))

ax3 = Axis(fig[2, 1:2], title = "Parameter convergence", xlabel = "Iteration",
           ylabel = "Relative change ensemble variance", yscale = log10)

for (p, name) in enumerate(free_parameters.names)
    θp_variances = [θ_variances[iter][p] for iter = 1:eki.iteration]
    lines!(ax3, θp_variances / θp_variances[1], label = String(name), linewidth = 2)
end

axislegend(ax3, position = :rt)

##display(fig)

save("perfect_catke_calibration_summary.svg", fig); nothing #hide

# ![](perfect_catke_calibration_summary.svg)

final_mean_θ = eki.iteration_summaries[end].ensemble_mean
#forward_run!(calibration, [θ★, final_mean_θ])
forward_run!(calibration, θ★)

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
b★ = interior(b)[1, 1, :]
b¹ = interior(b)[2, 1, :]

e★ = interior(e)[1, 1, :]
e¹ = interior(e)[2, 1, :]

u★ = interior(u)[1, 1, :]
u¹ = interior(u)[2, 1, :]

v★ = interior(v)[1, 1, :]
v¹ = interior(v)[2, 1, :]

fig = Figure()

ax = Axis(fig[1, 1], xlabel = "Buoyancy [m s⁻²]", ylabel = "z [m]")
b★_label = "true b at t = " * prettytime(t)
b¹_label = "b with ⟨θ⟩"
lines!(ax, b★, z; label=b★_label, linewidth=2)
lines!(ax, b¹, z; label=b¹_label, linewidth=2)
axislegend(ax, position=:lt)

ax = Axis(fig[1, 2], xlabel = "Turbulent kinetic energy [m² s⁻²]", ylabel = "z [m]")
e★_label = "true e at t = " * prettytime(t)
e¹_label = "e with ⟨θ⟩"
lines!(ax, e★, z; label=e★_label, linewidth=2)
lines!(ax, e¹, z; label=e¹_label, linewidth=2)
axislegend(ax, position=:lt)

ax = Axis(fig[1, 3], xlabel = "Turbulent kinetic energy [m² s⁻²]", ylabel = "z [m]")
u★_label = "true u at t = " * prettytime(t)
u¹_label = "u with ⟨θ⟩"
v★_label = "true v"
v¹_label = "v with ⟨θ⟩"
lines!(ax, u★, z; label=u★_label, linewidth=2)
lines!(ax, u¹, z; label=u¹_label, linewidth=2)
lines!(ax, v★, z; label=v★_label, linestyle=:dash, linewidth=2)
lines!(ax, v¹, z; label=v¹_label, linestyle=:dash, linewidth=2)
axislegend(ax, position=:lt)

save("perfect_catke_calibration_particle_realizations.svg", fig); nothing # hide

# ![](perfect_catke_calibration_particle_realizations.svg)

##display(fig)

# And also we plot the the distributions of the various model ensembles for few EKI iterations to see
# if and how well they converge to the true diffusivity values.

fig = Figure()

ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[2, 1], xlabel = "Cᴷu⁻ [m² s⁻¹]", ylabel = "Cᴷc⁻ [m² s⁻¹]")
ax3 = Axis(fig[2, 2])
scatters = []

for iteration in [1, 2, 3, 11]
    ## Make parameter matrix
    parameters = eki.iteration_summaries[iteration].parameters
    Nensemble = length(parameters)
    parameter_ensemble_matrix = [parameters[i][j] for i=1:Nensemble, j=1:2]

    push!(scatters, scatter!(ax2, parameter_ensemble_matrix))
    density!(ax1, parameter_ensemble_matrix[:, 1])
    density!(ax3, parameter_ensemble_matrix[:, 2], direction = :y)
end

vlines!(ax1, [θ★.Cᴷu⁻], color = :red)
vlines!(ax2, [θ★.Cᴷu⁻], color = :red)
hlines!(ax2, [θ★.Cᴷc⁻], color = :red)
hlines!(ax3, [θ★.Cᴷc⁻], color = :red)

colsize!(fig.layout, 1, Fixed(300))
colsize!(fig.layout, 2, Fixed(200))
rowsize!(fig.layout, 1, Fixed(200))
rowsize!(fig.layout, 2, Fixed(300))

Legend(fig[1, 2], scatters, ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration 10"],
       position = :lb)

hidedecorations!(ax1, grid = false)
hidedecorations!(ax3, grid = false)

##display(fig)

save("perfect_catke_calibration_parameter_distributions.svg", fig); nothing # hide

# ![](perfect_catke_calibration_parameter_distributions.svg)
