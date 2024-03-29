module EnsembleSimulations

export ensemble_column_model_simulation

using DataDeps

using ..Utils: tupleit
using ..Observations: SyntheticObservations, batch, tupleit

using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.Architectures: arch_array

function ensemble_column_model_simulation(observations;
                                          closure,
                                          Nensemble,
                                          Δt = 1.0,
                                          verbose = true,
                                          architecture = CPU(),
                                          tracers = :b,
                                          buoyancy = BuoyancyTracer(),
                                          non_ensemble_closure = nothing,
                                          kwargs...)

    observations = batch(observations)
    Nbatch = length(observations)

    # Assuming all observations are on similar grids...
    Nz = first(observations).grid.Nz
    Hz = first(observations).grid.Hz
    Lz = first(observations).grid.Lz

    column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(Nensemble, Nbatch))
    column_ensemble_halo_size = ColumnEnsembleSize(Nz=0, Hz=Hz)

    grid = RectilinearGrid(architecture,
                           size = column_ensemble_size,
                           halo = column_ensemble_halo_size,
                           topology = (Flat, Flat, Bounded),
                           z = (-Lz, 0))

    coriolis_ensemble = [FPlane(f=observations[j].metadata.coriolis.f) for i = 1:Nensemble, j=1:Nbatch]
    coriolis_ensemble = arch_array(architecture, coriolis_ensemble)

    closure_ensemble = [deepcopy(closure) for i = 1:Nensemble, j=1:Nbatch]
    closure_ensemble = arch_array(architecture, closure_ensemble)

    if isnothing(non_ensemble_closure)
        closure = closure_ensemble
    else
        non_ensemble_closure = tupleit(non_ensemble_closure)
        closure = (closure_ensemble, non_ensemble_closure...)
    end

    momentum_boundary_conditions =
        (; u = FieldBoundaryConditions(top = FluxBoundaryCondition(zeros(grid, Nensemble, Nbatch))))

    ensemble_tracer_bcs() = FieldBoundaryConditions(top = FluxBoundaryCondition(zeros(grid, Nensemble, Nbatch)),
                                                    bottom = GradientBoundaryCondition(zeros(grid, Nensemble, Nbatch)))

    tracers = tupleit(tracers)
    tracer_boundary_conditions = NamedTuple(name => ensemble_tracer_bcs() for name in tracers if name != :e)

    boundary_conditions = merge(momentum_boundary_conditions, tracer_boundary_conditions)

    ensemble_model = HydrostaticFreeSurfaceModel(; grid, tracers, buoyancy, boundary_conditions, closure,
                                                 coriolis = coriolis_ensemble,
                                                 kwargs...)

    ensemble_simulation = Simulation(ensemble_model; Δt, verbose, stop_time=first(observations).times[end])

    return ensemble_simulation
end

end # module
