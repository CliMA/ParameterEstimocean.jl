using Oceananigans
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: VerticalFormulation, HorizontalFormulation

using ParameterEstimocean
using ParameterEstimocean.Observations: FieldTimeSeriesCollector

using Statistics
using GLMakie

import Oceananigans.TurbulenceClosures: viscosity, diffusivity

struct ConstantHorizontalTracerDiffusivity <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, HorizontalFormulation}
    κh :: Float64
end

struct ConstantVerticalTracerDiffusivity <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, VerticalFormulation}
    κz :: Float64
end

ConstantHorizontalTracerDiffusivity(; κh=0.0) = ConstantHorizontalTracerDiffusivity(κh)
ConstantVerticalTracerDiffusivity(; κz=0.0) = ConstantVerticalTracerDiffusivity(κz)

@inline viscosity(::ConstantHorizontalTracerDiffusivity, args...) = 0.0
@inline diffusivity(closure::ConstantHorizontalTracerDiffusivity, args...) = closure.κh

@inline viscosity(::ConstantVerticalTracerDiffusivity, args...) = 0.0
@inline diffusivity(closure::ConstantVerticalTracerDiffusivity, args...) = closure.κz

function random_simulation(size=(16, 16, 16))
    grid = RectilinearGrid(; size, extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))

    closure = (ConstantHorizontalTracerDiffusivity(1.0), ConstantVerticalTracerDiffusivity(0.5))

    model = HydrostaticFreeSurfaceModel(; grid, closure,
                                        tracer_advection = nothing,
                                        velocities = PrescribedVelocityFields(),
                                        tracers = :c,
                                        buoyancy = nothing)

    simulation = Simulation(model, Δt=1e-3, stop_iteration=10)

    return simulation
end

test_simulation = random_simulation()

model = test_simulation.model
test_simulation.output_writers[:d3] = JLD2OutputWriter(model, model.tracers,
                                                       schedule = IterationInterval(10),
                                                       filename = "random_simulation_fields",
                                                       overwrite_existing = true)

slice_indices = (1, :, :)
test_simulation.output_writers[:d2] = JLD2OutputWriter(model, model.tracers,
                                                       schedule = IterationInterval(10),
                                                       filename = "random_simulation_slices",
                                                       indices = slice_indices,
                                                       overwrite_existing = true)

cᵢ(x, y, z) = randn()
set!(model, c=cᵢ)
run!(test_simulation)

simulation_ensemble = [random_simulation() for _ = 1:10]
times = [0.0, time(test_simulation)]

priors = (κh = ScaledLogitNormal(bounds=(0.0, 2.0)),
          κz = ScaledLogitNormal(bounds=(0.0, 2.0)))
          
free_parameters = FreeParameters(priors) 
observations = SyntheticObservations("random_simulation_slices.jld2"; field_names=:c, times)

c₀ = FieldTimeSeries("random_simulation_fields.jld2", "c")[1]

function initialize_simulation!(sim, parameters)
    c = sim.model.tracers.c
    set!(c, c₀)


    return 
end

function slice_collector(sim)
    c = sim.model.tracers.c
    c_slice = Field(c; indices=slice_indices)
    return FieldTimeSeriesCollector((; c=c_slice), times)
end

time_series_collector_ensemble = [slice_collector(sim) for sim in simulation_ensemble]

ip = InverseProblem(observations, simulation_ensemble, free_parameters;
                    time_series_collector = time_series_collector_ensemble,
                    initialize_with_observations = false,
                    initialize_simulation = initialize_simulation!)

#random_κ = [(; κh=10rand(), κz=10rand()) for sim in simulation_ensemble]
#G = forward_map(ip, random_κ, suppress=false)

eki = EnsembleKalmanInversion(ip; pseudo_stepping=ConstantConvergence(0.2))
iterate!(eki, iterations=10)

fig = Figure()
ax = Axis(fig[1, 1])

for iter in [0, 1, 4, 10]
    summary = eki.iteration_summaries[iter]
    κh = map(θ -> θ.κh, summary.parameters)
    κz = map(θ -> θ.κz, summary.parameters)
    scatter!(ax, κh, κz, label="Iteration $iter")
end

axislegend(ax)

display(fig)

