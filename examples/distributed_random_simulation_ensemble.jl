using Oceananigans
using Oceananigans.TurbulenceClosures: AbstractScalarDiffusivity, ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures: VerticalFormulation, HorizontalFormulation
using LinearAlgebra

using ParameterEstimocean
using ParameterEstimocean.Observations: FieldTimeSeriesCollector

using Statistics
#using GLMakie

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

#=
c₀ = FieldTimeSeries("random_simulation_fields.jld2", "c")[1]
c_slices = FieldTimeSeries("random_simulation_slices.jld2", "c")
c_averaged_slices = FieldTimeSeries("random_simulation_averaged_slices.jld2", "c")

@show c_slices[end]
@show c_averaged_slices[end]
=#

#####
##### On all ranks
#####

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
    #return FieldTimeSeriesCollector((; c=c_slice), times)
    return FieldTimeSeriesCollector((; c=c_slice), times, averaging_window=1e-1)
end

simulation = random_simulation()
time_series_collector = slice_collector(simulation)

ip = InverseProblem(observations, [simulation], free_parameters;
                    time_series_collector = [time_series_collector],
                    initialize_with_observations = false,
                    initialize_simulation = initialize_simulation!)

#random_κ = [(; κh=10rand(), κz=10rand()) for sim in simulation_ensemble]
θ = [(; κh=10rand(), κz=10rand())]
X = transform_to_unconstrained(ip.free_parameters.priors, θ)
#θ = transform_to_constrained(ip.free_parameters.priors, X)
G = inverting_forward_map(ip, X, suppress=false)

#####
##### Next...
#####

# 1. Compute y, Γy on every rank.
y = observation_map(ip)
Nobs = length(y)
Γy = Matrix(I, Nobs, Nobs)

# 2. Create random parameters X on every rank. size(X) = (Nparams, Nranks)
X = random_unconstrained_parameters(ip.free_parameters, Nranks)

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
