module EnsembleSimulations

using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.Architectures: arch_array

function ensemble_column_model_simulation(observations;
                                          closure,
                                          Nensemble,
                                          Δt = 1.0,
                                          architecture = CPU(),
                                          tracers = :b,
                                          buoyancy = BuoyancyTracer(),
                                          kwargs...)

    observations isa Vector || (observations = [observations]) # Singleton batch
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

    Qᵘ = zeros(grid, Nensemble, Nbatch)
    Qᵇ = zeros(grid, Nensemble, Nbatch)
    N² = zeros(grid, Nensemble, Nbatch)

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))

    ensemble_model = HydrostaticFreeSurfaceModel(; grid, tracers, buoyancy,
                                                 boundary_conditions = (; u=u_bcs, b=b_bcs),
                                                 coriolis = coriolis_ensemble,
                                                 closure = closure_ensemble,
                                                 kwargs...)

    ensemble_simulation = Simulation(ensemble_model; Δt, stop_time=first(observations).times[end])

    return ensemble_simulation
end

end # module

