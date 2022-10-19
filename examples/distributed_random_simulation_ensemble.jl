using Oceananigans
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: VerticalFormulation, HorizontalFormulation
using LinearAlgebra

using ParameterEstimocean
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

function random_simulation(size=(16, 16, 16))
    grid = RectilinearGrid(; size, extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))

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

simulation = random_simulation()
time_series_collector = slice_collector(simulation)

ip = InverseProblem(observations, simulation, free_parameters;
                    time_series_collector = time_series_collector,
                    initialize_with_observations = false,
                    initialize_simulation = initialize_simulation!)

#random_parameters = (κh=rand(), κz=rand())
#G = forward_map(ip, random_parameters; suppress=false)

dip = DistributedInverseProblem(ip)

Random.seed!(123)
eki = EnsembleKalmanInversion(dip; pseudo_stepping=ConstantConvergence(0.3))
iterate!(eki; iterations=3)

#=
random_parameters = [(κh=rand(), κz=rand()) for i = 1:Nensemble(dip)]
@show random_parameters

G = forward_map(dip, random_parameters; suppress=false)

if rank == 0
    @show size(G) G
end
=#

#=
#####
##### Next...
#####

# 1. Compute y, Γy on every rank.
y = observation_map(ip)
Nobs = length(y)
Γy = Matrix(I, Nobs, Nobs)

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

# 4. Compute local G
G = forward_map(ip, local_parameters, suppress=false)

# 5. Gather G to global_G on rank 0
global_G = MPI.Gather(G, 0, comm)
MPI.Barrier(comm)

@show global_G

# EKI step on rank 0
# if rank == 0
    # global_G = reshape(global_G, Nobs, nproc)
    # I don't know how to do the EKI step
    # unconstrained_parameters = eki_step
# end

# go back to step 3.


MPI.Finalize()
# forward_map_output = global_G = zeros(length(y), Nranks)
# pseudo_stepping = ConstantConvergence(0.2)
# eki = EnsembleKalmanInversion(ip;
#                               pseudo_stepping,
#                               forward_map_output,
#                               unconstrained_parameters,
#                               Nensemble=Nranks)

# 3. Compute G for every rank with inverting_forward_map(ip, X[rank:rank, :])
#
#        -> use "one ensemble member InverseProblem" on each rank
# 4. All-to-all to build "global_G". size(global_G) = (Nobs, Nranks)
# 5. new_X = adaptive_step_parameters(pseudostepping, X, global_G, y, Γy)
# 6. Repeat 3-5.


#=
eki = EnsembleKalmanInversion(ip; pseudo_stepping=ConstantConvergence(0.3))
iterate!(eki, iterations=10)

@show eki.iteration_summaries[end]

fig = Figure()
ax = Axis(fig[1, 1])

for iter in 0:10
    summary = eki.iteration_summaries[iter]
    κh = map(θ -> θ.κh, summary.parameters)
    κz = map(θ -> θ.κz, summary.parameters)
    scatter!(ax, κh, κz, label="Iteration $iter")
end

axislegend(ax)

display(fig)
=#
=#
