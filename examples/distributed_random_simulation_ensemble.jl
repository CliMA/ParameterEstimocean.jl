using Oceananigans
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: VerticalFormulation, HorizontalFormulation
using LinearAlgebra

using ParameterEstimocean
using ParameterEstimocean.Utils: map_gpus_to_ranks!
using ParameterEstimocean.Observations: FieldTimeSeriesCollector
using ParameterEstimocean.Parameters: random_unconstrained_parameters
using ParameterEstimocean.InverseProblems: Nensemble

using Statistics
using MPI

import Oceananigans.TurbulenceClosures: viscosity, diffusivity

using Random

MPI.Init()
comm = MPI.COMM_WORLD

rank  = MPI.Comm_rank(comm)
nproc = MPI.Comm_size(comm)

arch = GPU()

if arch isa GPU
    map_gpus_to_ranks!()
end

@show rank nproc

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

stop_time = 1e-1

function random_simulation(arch; size=(4, 4, 4))
    grid = RectilinearGrid(arch; size, extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))

    closure = (ConstantHorizontalTracerDiffusivity(1.0), ConstantVerticalTracerDiffusivity(0.5))

    model = HydrostaticFreeSurfaceModel(; grid, closure,
                                        tracer_advection = nothing,
                                        velocities = PrescribedVelocityFields(),
                                        tracers = :c,
                                        buoyancy = nothing)

    simulation = Simulation(model; Δt=1e-3, stop_time)

    return simulation
end

#####
##### On rank 0
#####

if rank == 0
    test_simulation = random_simulation(arch)

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
end

# Finished simulation, wait rank 0
MPI.Barrier(comm)

@show rank

#####
##### On all ranks
#####

slice_indices = (1, :, :)

times = [0.0, stop_time]

priors = (κh = ScaledLogitNormal(bounds=(0.0, 2.0)),
          κz = ScaledLogitNormal(bounds=(0.0, 2.0)))
          
free_parameters = FreeParameters(priors) 
#obspath = "random_simulation_slices.jld2"
obspath = "random_simulation_averaged_slices.jld2"
observations = SyntheticObservations(obspath; field_names=:c, times)

# Initial condition

function initialize_simulation!(sim, parameters)
    c₀ = FieldTimeSeries("random_simulation_fields.jld2", "c")[1]
    c = sim.model.tracers.c
    set!(c, c₀)
    return nothing
end

function slice_collector(sim)
    c = sim.model.tracers.c
    c_slice = Field(c; indices=slice_indices)
    return FieldTimeSeriesCollector((; c=c_slice), times, averaging_window=1e-1)
end

simulation = random_simulation(arch)
time_series_collector = slice_collector(simulation)

ip = InverseProblem(observations, simulation, free_parameters;
                    time_series_collector = time_series_collector,
                    initialize_with_observations = false,
                    initialize_simulation = initialize_simulation!)

dip = DistributedInverseProblem(ip)

Random.seed!(123)
eki = EnsembleKalmanInversion(dip; pseudo_stepping=ConstantConvergence(0.3))

iterate!(eki; iterations=10)

θ̅(iteration) = [eki.iteration_summaries[iteration].ensemble_mean...]
varθ(iteration) = eki.iteration_summaries[iteration].ensemble_var

@show θ̅(9) varθ(9)

#=
#####
##### To do everything on rank 0
#####

Nθ = length(ip.free_parameters.names)

# 2. Create parameters on rank 0
if rank == 0
    unconstrained_parameters = random_unconstrained_parameters(ip.free_parameters, nproc)
else
    unconstrained_parameters = Float64[]
end

MPI.Barrier(comm)

# 3. Scatter parameters
local_parameters = MPI.Scatter(unconstrained_parameters, Nθ, 0, comm)
@show rank, unconstrained_parameters, local_parameters

=#
