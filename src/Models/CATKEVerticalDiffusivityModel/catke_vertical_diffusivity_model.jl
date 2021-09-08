using Oceananigans.Architectures: arch_array

"""
    EnsembleModel(data_batch::OneDimensionalTimeSeriesObservations; architecture = CPU(), N_ens = 50, kwargs...)

Build an Oceananigans `HydrostaticFreeSurfaceModel` with many independent 
columns. The model grid is given by the data in `data_batch`, and the
dynamics in each column is attached to its own `CATKEVerticalDiffusivity` closure stored in 
an `(Nx, Ny)` Matrix of closures. `N_ens = 50` is the 
desired count of ensemble members for calibration with Ensemble Kalman Inversion (EKI), and the 
remaining keyword arguments `kwargs` define the default closure across all columns.

In the "many columns" configuration, we run the model on a 3D grid with `(Flat, Flat, Bounded)` boundary 
conditions so that many independent columns can be evolved at once with much of the computational overhead
split among the columns. The `Nx` rows of vertical columns are each reserved for an "ensemble" member
whose attached parameter value (updated at each iteration of EKI) sets the diffusivity closure
used to predict the model solution for the `Ny` physical scenarios described by the simulation-specific 
`OneDimensionalTimeSeries` objects in `data_batch`.
"""
function EnsembleModel(data_batch::OneDimensionalTimeSeriesObservations; architecture = CPU(), N_ens = 50, kwargs...)

    data_grid = data_batch[1].grid
    grid = ColumnEnsembleGrid(data_grid; size=(N_ens, length(data_batch), data_grid.Nz))

    closure = [CATKEVerticalDiffusivity(Float64; warning=false, kwargs...) for i=1:N_ens, j=1:length(data_batch)]

    coriolis = [FPlane(f=td.constants[:f]) for i=1:N_ens, td in data_batch]
    coriolis = arch_array(architecture, coriolis)

    bc_matrix(f) = [f(td.boundary_conditions) for i = 1:N_ens, td in data_batch]
    Qᵇ = bc_matrix(bc -> bc.Qᵇ)
    Qᵘ = bc_matrix(bc -> bc.Qᵘ)
    Qᵛ = bc_matrix(bc -> bc.Qᵛ)
    dbdz_bottom = bc_matrix(bc -> bc.dbdz_bottom)
    dudz_bottom = bc_matrix(bc -> bc.dudz_bottom)

    # Convert to CuArray if necessary
    closure = arch_array(architecture, closure)
    Qᵇ = arch_array(architecture, Qᵇ)
    Qᵘ = arch_array(architecture, Qᵘ)
    Qᵛ = arch_array(architecture, Qᵛ)
    dbdz_bottom = arch_array(architecture, dbdz_bottom)
    dudz_bottom = arch_array(architecture, dudz_bottom)
    
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ), 
                                    bottom = GradientBoundaryCondition(dudz_bottom))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), 
                                    bottom = GradientBoundaryCondition(dbdz_bottom))

    model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                         grid = grid,
                                         tracers = (:b, :e),
                                         buoyancy = BuoyancyTracer(),
                                         coriolis = coriolis,
                                         boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                         closure = closure)

    return model
end
