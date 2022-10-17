using Oceananigans
using ParameterEstimocean
using Statistics

function random_simulation(size=(16, 16, 16))
    grid = RectilinearGrid(; size, extent=(2π, 2π, 2π), topology=(Periodic, Periodic, Periodic))

    model = HydrostaticFreeSurfaceModel(; grid,
                                        tracer_advection = nothing,
                                        velocities = PrescribedVelocityFields(),
                                        tracers = :c,
                                        buoyancy = nothing,
                                        closure = ScalarDiffusivity(κ=1.0))

    simulation = Simulation(model, Δt=1e-3, stop_iteration=10)

    return simulation
end

test_simulation = random_simulation()

model = test_simulation.model
test_simulation.output_writers[:c] = JLD2OutputWriter(model, model.tracers,
                                                      schedule = IterationInterval(10),
                                                      filename = "random_simulation",
                                                 overwrite_existing = true)

run!(test_simulation)

simulation_ensemble = [random_simulation() for _ = 1:10]
times = [0.0, time(test_simulation)]
priors = (; c = ScaledLogitNormal(bounds=(0.0, 10.0)))
free_parameters = FreeParameters(priors) 
observations = SyntheticObservations("random_simulation.jld2"; field_names=:c, times)
ip = InverseProblem(observations, simulation_ensemble, free_parameters)

#cᵢ = rand(size(grid)...)
#set!(model, c=cᵢ)

