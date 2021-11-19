# # Intro to `InverseProblem`
#
# This example illustrates the construction of an "ensemble simulation" that can
# evaluate an ensemble of column models given an ensemble of free parameter sets.
# The example then builds an `InverseProblem` from observations, an ensemble simulation,
# and a set of free parameters, and illustrates its basic usage.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, CairoMakie"
# ```

# First we load few things

using OceanTurbulenceParameterEstimation

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

using CairoMakie
using Distributions
using JLD2

# We reuse some utilities from a previous example to build observations:

examples_path = joinpath(pathof(OceanTurbulenceParameterEstimation), "..", "..", "examples")
include(joinpath(examples_path, "intro_to_observations.jl"))
data_path = generate_synthetic_observations()
observations = OneDimensionalTimeSeries(data_path, field_names=:b, normalize=ZScore)

# # Building an "ensemble simulation"
#
# Our next task is to construct a parameterized `Oceananigans.Simulation`
# that generates the "foward map" for an ensemble of free parameter sets.
# To generate an ensemble of column model model outputs efficiently, we construct one
# 3D `Oceananigans.Simulation` consisting of `Nx` by `Ny` independent column models.
#
# The calibration problem then uses the ensemble
# simulation to find optimal parameters by minimizing the discrepency between
# the observations and the forward map.

"""
    extract_parameters(observations, Nensemble)

Extract parameters from a batch of observations.
"""
function extract_parameters(observations, Nensemble)
    Nbatch = length(observations)
    Qᵘ, Qᵇ, N², f = [zeros(Nensemble, Nbatch) for i = 1:4]

    Nz = first(observations).grid.Nz
    Hz = first(observations).grid.Hz
    Lz = first(observations).grid.Lz
    Δt = first(observations).metadata.Δt

    for (j, obs) in enumerate(observations)
        Qᵘ[:, j] .= obs.metadata.parameters.Qᵘ
        Qᵇ[:, j] .= obs.metadata.parameters.Qᵇ
        N²[:, j] .= obs.metadata.parameters.N²
        f[:, j] .= obs.metadata.coriolis.f
    end

    return Qᵘ, Qᵇ, N², f, Δt, Lz, Nz, Hz
end

"""
    build_ensemble_simulation(observations; Nensemble=1)

Returns an `Oceananigans.Simulation` representing an `Nensemble × 1`
ensemble of column models designed to reproduce `observations`.
"""
function build_ensemble_simulation(observations; Nensemble=1)

    observations isa Vector || (obserations = [observations]) # Singleton batch
    Nbatch = length(observations)

    Qᵘ, Qᵇ, N², f, Δt, Lz, Nz, Hz = extract_parameters(observations)

    column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(Nensemble, Nbatch), Hz=Hz)
    ensemble_grid = RectilinearGrid(size = column_ensemble_size, topology = (Flat, Flat, Bounded), z = (-Lz, 0))

    coriolis_ensemble = [FPlane(f=f[i, j]) for i = 1:Nensemble, j=1:Nbatch]
    closure_ensemble = [deepcopy(closure) for i = 1:Nensemble, j=1:Nbatch]

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))
                        
    ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                                 tracers = :b,
                                                 buoyancy = BuoyancyTracer(),
                                                 boundary_conditions = (; u=u_bcs, b=b_bcs),
                                                 coriolis = coriolis_ensemble,
                                                 closure = closure_ensemble)

    ensemble_simulation = Simulation(ensemble_model; Δt=Δt, stop_time=observations.times[end])

    return ensemble_simulation, optimal_closure
end

# The following illustrations uses a simple ensemble simulation with two ensemble members:

ensemble_simulation, closure★ = build_ensemble_simulation(observations; Nensemble=3)

# # Free parameters
#
# We construct some prior distributions for our free parameters. We found that it often helps to
# constrain the prior distributions so that neither very high nor very low values for diffusivities
# can be drawn out of the distribution.

priors = (convective_κz = lognormal_with_mean_std(0.3, 0.5),
          background_κz = lognormal_with_mean_std(2.5e-4, 0.25e-4))

free_parameters = FreeParameters(priors)

# We also take the opportunity to collect a named tuple of the optimal parameters

θ★ = (convective_κz = closure★.convective_κz,
      background_κz = closure★.background_κz)

# ## Visualizing the priors
#
# We visualize our prior distributions by plotting a huge number
# of samples:

using OceanTurbulenceParameterEstimation.EnsembleKalmanInversions: convert_prior, inverse_parameter_transform

Nsamples = 50000000

samples(prior) = [inverse_parameter_transform(prior, θ) for θ in rand(convert_prior(prior), Nsamples)]

convective_κz_samples = samples(priors.convective_κz)
background_κz_samples = samples(priors.background_κz)

fig = Figure()
ax_top = Axis(fig[1, 1], xlabel = "convective κᶻ [m² s⁻¹]", ylabel = "Density")
density!(ax_top, convective_κz_samples)
xlims!(ax_top, 0, 10)

ax_bottom = Axis(fig[2, 1], xlabel = "background κᶻ [m² s⁻¹]", ylabel = "Density")
density!(ax_bottom, background_κz_samples)

save("prior_visualization.svg", fig)
nothing # hide

# ![](prior_visualization.svg)

# # The `InverseProblem`

# We can construct the inverse problem ``y = G(θ) + η``. Here, ``y`` are the `observations` and ``G`` is the
# `ensemble_model`.

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# ## Using `InverseProblem` to compute `forward_map`
#
# As a sanity check we apply the `forward_map` on the calibration after we initialize all ensemble
# members with the true parameter values. We then confirm that the output of the `forward_map` matches
# the observations to machine precision.

θ¹ = (convective_κz = 0.8 * θ★.convective_κz,
      background_κz = 9.0 * θ★.background_κz)

θ² = (convective_κz = 2.0 * θ★.convective_κz,
      background_κz = 0.1 * θ★.background_κz)

θ_ensemble = [θ★, θ¹, θ²]

G = forward_map(calibration, θ_ensemble)
y = observation_map(calibration)

# The `forward_map` output `G` is a two-dimensional matrix whose first dimension
# is the size of the state space and whose second dimension is the `ensemble_size`.
# Here we ensure that mapped output first ensemble member, which was run with the "true"
# parameters, is identical to the mapped observations:

G[:, 1] ≈ y

# Visualizing forward model output
#
# Next we visualize the discrepency between solutions generated by true
# and non-optimal parameter sets `θ¹` and `θ²`. Time-series data from
# the ensemble run is collected by `calibration.time_series_collector`:

time_series_collector = calibration.time_series_collector
times = time_series_collector.times

## Extract last save point and plot each solution component
Nt = length(times)

b = time_series_collector.field_time_serieses.b[Nt]
t = times[Nt]
z = znodes(b)

## The ensemble varies along the first, or `x`-dimension:
b★ = interior(b)[1, 1, :]
b¹ = interior(b)[2, 1, :]
b² = interior(b)[3, 1, :]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Buoyancy [m s⁻²]")

b★_label = "true b at t = " * prettytime(t)
b¹_label = "b with $θ¹"
b²_label = "b with $θ²"

lines!(ax, b★, z; label=b★_label)
lines!(ax, b¹, z; label=b¹_label)
lines!(ax, b², z; label=b²_label)

axislegend(ax, position=:rb)

save("ensemble_simulation_demonstration.svg", fig)
nothing # hide

# ![](ensemble_simulation_demonstration.svg)

