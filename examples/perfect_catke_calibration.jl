# # Perfect CAKTE calibration with Ensemble Kalman Inversion

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add ParameterEstimocean, Oceananigans, Distributions, CairoMakie"
# ```

using ParameterEstimocean, LinearAlgebra, CairoMakie

using ParameterEstimocean.Transformations: Transformation
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities:
    CATKEVerticalDiffusivity, MixingLength, TurbulentKineticEnergyEquation

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
examples_path = joinpath(pathof(ParameterEstimocean), "..", "..", "examples")
include(joinpath(examples_path, "intro_to_inverse_problems.jl"))

mixing_length = MixingLength(Cᴬc  = 0.5,
                             Cᴬe  = 0.5,
                             Cᵇu  = 0.5,
                             Cᵇc  = 0.5,
                             Cᵇe  = 0.5,
                             Cᴷu⁻ = 0.5,
                             Cᴷc⁻ = 0.5,
                             Cᴷe⁻ = 0.5)

turbulent_kinetic_energy_equation = TurbulentKineticEnergyEquation(CᴰRiʷ=0.5, Cᵂu★=0.5, CᵂwΔ=0.5, Cᴰ⁻=0.5)

catke = CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation)

@show catke

## Specify both wind mixing and convection:
Δt = 1e-16
data_path = generate_synthetic_observations("catke",
                                            closure = catke,
                                            tracers = (:b, :e),
                                            Nz = 32,
                                            Lz = 64,
                                            Δt = Δt,
                                            output_schedule = IterationInterval(1),
                                            stop_time = Δt,
                                            overwrite = true,
                                            Qᵘ = -1e-4,
                                            Qᵇ = 1e-8,
                                            N² = 1e-5)

# Next, we load and inspect the observations to make sure they're sensible:

transformation = (u = ZScore(),
                  v = ZScore(),
                  b = ZScore(),
                  e = RescaledZScore(1e-1))

observations = SyntheticObservations(data_path; field_names=(:u, :v, :b, :e), transformation)

fig = Figure()

ax_b = Axis(fig[1, 1], xlabel = "Buoyancy\n[10⁻⁴ m s⁻²]", ylabel = "z [m]")
ax_u = Axis(fig[1, 2], xlabel = "Velocities\n[cm s⁻¹]")
ax_e = Axis(fig[1, 3], xlabel = "Turbulent kinetic energy\n[cm² s⁻²]")

z = znodes(observations.grid, Center())

colorcycle = [:black, :red, :blue, :orange, :pink]

#for i = 1:length(observations.times)
i = 2
    b = observations.field_time_serieses.b[i]
    e = observations.field_time_serieses.e[i]
    u = observations.field_time_serieses.u[i]
    v = observations.field_time_serieses.v[i]
    t = observations.times[i]

    label = "t = " * prettytime(t)
    u_label = i == 1 ? "u, " * label : label
    v_label = i == 1 ? "v, " * label : label
    ## Note unit conversions below, eg m s⁻² -> 10⁻⁴ m s⁻²
    lines!(ax_b, 1e4 * interior(b)[1, 1, :], z; label, color=colorcycle[i])
    lines!(ax_u, 1e2 * interior(u)[1, 1, :], z; linestyle=:solid, color=colorcycle[i], label=u_label)
    lines!(ax_u, 1e2 * interior(v)[1, 1, :], z; linestyle=:dash, color=colorcycle[i], label=v_label)
    lines!(ax_e, 1e4 * interior(e)[1, 1, :], z; label, color=colorcycle[i])
#end

axislegend(ax_b, position=:rb)
axislegend(ax_u, position=:lb, merge=true)
axislegend(ax_e, position=:rb)

display(fig)

#=
save("synthetic_catke_observations.svg", fig); nothing # hide

# ![](synthetic_catke_observations.svg)

# Well, that looks like a boundary layer, in some respects.
# 
# # Calibration
#
# Next, we build a simulation of an ensemble of column models to calibrate
# CATKE using Ensemble Kalman Inversion.

architecture = CPU()
ensemble_simulation, closure★ = build_ensemble_simulation(observations, architecture; Nensemble=20)

# We choose to calibrate a subset of the CATKE parameters,

priors = (Cᴬu = lognormal(mean=0.5, std=0.1),
          Cᴬc = lognormal(mean=0.5, std=0.1),
          Cᴬe = lognormal(mean=0.5, std=0.1))

free_parameters = FreeParameters(priors)

# The handy utility function `build_ensemble_simulation` also tells us the optimal
# parameters that were used when generating the synthetic observations:

@show θ★ = (Cᴬu = closure★.mixing_length.Cᴬu,
            Cᴬc = closure★.mixing_length.Cᴬc,
            Cᴬe = closure★.mixing_length.Cᴬe)

# We construct the `InverseProblem` from `observations`, `ensemble_simulation`, and
# `free_parameters`,

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# We can check that the first ensemble member of the mapped output, which was run with
# the "true" # parameters, is identical to the mapped observations:

G = forward_map(calibration, θ★)
y = observation_map(calibration)

@show G[:, 1] ≈ y

# # Ensemble Kalman Inversion
#
# Next, we construct an `EnsembleKalmanInversion` (EKI) object,
#
# The calibration is done here using Ensemble Kalman Inversion. For more information about the 
# algorithm refer to
# [EnsembleKalmanProcesses.jl documentation](
# https://clima.github.io/EnsembleKalmanProcesses.jl/stable/ensemble_kalman_inversion/).

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 1e-3,
                              resampler = Resampler(acceptable_failure_fraction=0.3))

# and perform few iterations to see if we can converge to the true parameter values.

iterate!(eki; iterations = 10)

# Last, we visualize the outputs of EKI calibration.

## Convert everything to a vector
optimal_θ = collect(values(θ★))
ensemble_mean_θ = map(summary -> collect(values(summary.ensemble_mean)), eki.iteration_summaries)
θ_variances = map(summary -> collect(values(summary.ensemble_var)), eki.iteration_summaries)

absolute_error = NamedTuple(name => map(θ -> θ[p] - θ★[p], ensemble_mean_θ) for (p, name) in enumerate(free_parameters.names))
relative_error = NamedTuple(name => abs.(absolute_error[name]) ./ θ★[name] for name in free_parameters.names)

output_distances = map(θ -> norm(forward_map(calibration, θ)[:, 1:1] - y), ensemble_mean_θ)

fig = Figure()

ax_error = Axis(fig[1, 1], title = "Parameter distance", xlabel = "Iteration", ylabel = "|⟨θₙ⟩ - θ★| / θ★")

for name in free_parameters.names
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

axislegend(ax3, valign = :top, halign = :right)

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

e★ = 1e4 * interior(e)[1, 1, :]  # convert units m² s⁻² -> cm² s⁻²
e¹ = 1e4 * interior(e)[2, 1, :]  # convert units m² s⁻² -> cm² s⁻²

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

ax = Axis(fig[1, 2], xlabel = "Turbulent kinetic energy\n[cm² s⁻²]")
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
ax2 = Axis(fig[2, 1], xlabel = "Cᴬu", ylabel = "Cᴬc")
ax3 = Axis(fig[2, 2])
scatters = []
labels = String[]

for iteration in [0, 1, 5, 10]
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

Legend(fig[1, 2], scatters, labels, valign = :bottom, halign = :left)

hidedecorations!(ax1, grid = false)
hidedecorations!(ax3, grid = false)

##display(fig)

save("perfect_catke_calibration_parameter_distributions.svg", fig); nothing # hide

# ![](perfect_catke_calibration_parameter_distributions.svg)

# Hint: if using a REPL or notebook, try
# `using Pkg; Pkg.add("ElectronDisplay"); using ElectronDisplay; display(fig)`
# To see the figure in a window.
=#
