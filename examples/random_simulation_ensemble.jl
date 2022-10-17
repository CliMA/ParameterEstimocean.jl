using Oceananigans
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ExplicitTimeDiscretization, ThreeDimensionalFormulation
import Oceananigans.TurbulenceClosures: viscosity, diffusivity

using ParameterEstimocean
using ParameterEstimocean.Observations: FieldTimeSeriesCollector
using Statistics

struct ConstantTracerDiffusivity <: AbstractScalarDiffusivity{ExplicitTimeDiscretization, ThreeDimensionalFormulation}
    κ :: Float64
end

ConstantTracerDiffusivity(; κ=0.0) = ConstantTracerDiffusivity(κ)

@inline viscosity(::ConstantTracerDiffusivity, args...) = 0.0
@inline diffusivity(closure::ConstantTracerDiffusivity, args...) = closure.κ

function random_simulation(size=(16, 16, 16))
    grid = RectilinearGrid(; size, extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracer_advection = nothing,
                                        velocities = PrescribedVelocityFields(),
                                        tracers = :c,
                                        buoyancy = nothing,
                                        closure = ConstantTracerDiffusivity(κ=1.0))

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

run!(test_simulation)

simulation_ensemble = [random_simulation() for _ = 1:10]
times = [0.0, time(test_simulation)]
priors = (; c = ScaledLogitNormal(bounds=(0.0, 10.0)))
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

random_κ = [(; κ=10rand()) for sim in simulation_ensemble]

forward_run!(ip, random_κ)

#eki = EnsembleKalmanInversion(ip; pseudo_stepping=ConstantConvergence(0.9))

#cᵢ = rand(size(grid)...)
#set!(model, c=cᵢ)

