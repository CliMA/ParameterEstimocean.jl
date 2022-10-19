using Oceananigans
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: VerticalFormulation, HorizontalFormulation

using ParameterEstimocean
using ParameterEstimocean.Observations: FieldTimeSeriesCollector

using Random
using Statistics

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

function random_simulation(size=(4, 4, 4))
    grid = RectilinearGrid(; size, extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))

    closure = (ConstantHorizontalTracerDiffusivity(1.0), ConstantVerticalTracerDiffusivity(0.5))

    model = HydrostaticFreeSurfaceModel(; grid, closure,
                                        tracer_advection = nothing,
                                        velocities = PrescribedVelocityFields(),
                                        tracers = :c,
                                        buoyancy = nothing)

    simulation = Simulation(model, Δt=1e-3, stop_time=1e-1)

    return simulation
end

test_simulation = random_simulation()

model = test_simulation.model
test_simulation.output_writers[:d3] = JLD2OutputWriter(model, model.tracers,
                                                       schedule = IterationInterval(100),
                                                       filename = "random_simulation_fields",
                                                       overwrite_existing = true)

slice_indices = (1, :, :)
test_simulation.output_writers[:slices] = JLD2OutputWriter(model, model.tracers,
                                                           schedule = AveragedTimeInterval(1e-1),
                                                           filename = "random_simulation_averaged_slices",
                                                           indices = slice_indices,
                                                           overwrite_existing = true)

test_simulation.output_writers[:avg] = JLD2OutputWriter(model, model.tracers,
                                                        schedule = TimeInterval(1e-1),
                                                        filename = "random_simulation_slices",
                                                        indices = slice_indices,
                                                        overwrite_existing = true)

cᵢ(x, y, z) = randn()
set!(model, c=cᵢ)
run!(test_simulation)

c₀ = FieldTimeSeries("random_simulation_fields.jld2", "c")[1]
c_slices = FieldTimeSeries("random_simulation_slices.jld2", "c")
c_averaged_slices = FieldTimeSeries("random_simulation_averaged_slices.jld2", "c")

@show c_slices[end]
@show c_averaged_slices[end]

simulation_ensemble = [random_simulation() for _ = 1:4]
times = [0.0, time(test_simulation)]

priors = (κh = ScaledLogitNormal(bounds=(0.0, 2.0)),
          κz = ScaledLogitNormal(bounds=(0.0, 2.0)))
          
free_parameters = FreeParameters(priors) 
#obspath = "random_simulation_slices.jld2"
obspath = "random_simulation_averaged_slices.jld2"
observations = SyntheticObservations(obspath; field_names=:c, times)

# Initial condition
c₀ = FieldTimeSeries("random_simulation_fields.jld2", "c")[1]

function initialize_simulation!(sim, parameters)
    c = sim.model.tracers.c
    set!(c, c₀)
    return nothing
end

function slice_collector(sim)
    c = sim.model.tracers.c
    c_slice = Field(c; indices=slice_indices)
    #return FieldTimeSeriesCollector((; c=c_slice), times)
    return FieldTimeSeriesCollector((; c=c_slice), times, averaging_window=1e-1)
end

time_series_collector_ensemble = [slice_collector(sim) for sim in simulation_ensemble]

ip = InverseProblem(observations, simulation_ensemble, free_parameters;
                    time_series_collector = time_series_collector_ensemble,
                    initialize_with_observations = false,
                    initialize_simulation = initialize_simulation!)

#θ = [(; κh=10rand(), κz=10rand()) for sim in simulation_ensemble]
#G = forward_map(ip, θ, suppress=false)

Random.seed!(123)
eki = EnsembleKalmanInversion(ip; pseudo_stepping=ConstantConvergence(0.3))
@show eki.unconstrained_parameters
@show eki.forward_map_output
