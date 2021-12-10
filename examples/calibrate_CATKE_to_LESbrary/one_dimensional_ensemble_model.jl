using Oceananigans
using Oceananigans.Architectures: arch_array
using Oceananigans: AbstractModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel

function get_parameter(observation, parameter_path)
    file = jldopen(observation.path)
    parameter = file[parameter_path]
    close(file)
    return parameter
end

"""
    OneDimensionalEnsembleModel(observations::SyntheticObservationsBatch; architecture = CPU(), ensemble_size = 50, kwargs...)

Build an Oceananigans `HydrostaticFreeSurfaceModel` with many independent 
columns. The model grid is given by the data in `observations`, and the
dynamics in each column is attached to its own `CATKEVerticalDiffusivity` closure stored in 
an `(Nx, Ny)` Matrix of closures. `ensemble_size = 50` is the 
desired count of ensemble members for calibration with Ensemble Kalman Inversion (EKI), and the 
remaining keyword arguments `kwargs` define the default closure across all columns.

In the "many columns" configuration, we run the model on a 3D grid with `(Flat, Flat, Bounded)` boundary 
conditions so that many independent columns can be evolved at once with much of the computational overhead
split among the columns. The `Nx` rows of vertical columns are each reserved for an "ensemble" member
whose attached parameter value (updated at each iteration of EKI) sets the diffusivity closure
used to predict the model solution for the `Ny` physical scenarios described by the simulation-specific 
`SyntheticObservations` objects in `observations`.
"""
function OneDimensionalEnsembleModel(observations;
    architecture = CPU(),
    ensemble_size = 50,
    closure = CATKEVerticalDiffusivity(Float64; warning = false)
)
    observations = vectorize(observations)

    Lz = get_parameter(observations[1], "grid/Lz")
    g = get_parameter(observations[1], "buoyancy/gravitational_acceleration")
    α = get_parameter(observations[1], "buoyancy/equation_of_state/α")

    Nx = ensemble_size
    Ny = length(observations)
    Nz = observations[1].grid.Nz
    column_ensemble_size = ColumnEnsembleSize(Nz = Nz, ensemble = (Nx, Ny), Hz = 1)

    ensemble_grid = RectilinearGrid(size = column_ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
    closure_ensemble = arch_array(architecture, [closure for i = 1:Nx, j = 1:Ny])

    get_Qᵇ(observation) = get_parameter(observation, "parameters/boundary_condition_θ_top") * α * g
    get_Qᵘ(observation) = get_parameter(observation, "parameters/boundary_condition_u_top")
    get_f₀(observation) = get_parameter(observation, "coriolis/f")
    get_dudz_bottom(observation) = get_parameter(observation, "parameters/boundary_condition_u_bottom")
    get_dbdz_bottom(observation) = get_parameter(observation, "parameters/boundary_condition_θ_bottom") * α * g

    to_arch_array(f) = arch_array(architecture, [f(observations[j]) for i = 1:Nx, j = 1:Ny])

    Qᵇ_ensemble = to_arch_array(get_Qᵇ)
    Qᵘ_ensemble = to_arch_array(get_Qᵘ)
    ensemble_u_bottom = to_arch_array(get_dudz_bottom)
    ensemble_b_bottom = to_arch_array(get_dbdz_bottom)

    ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ_ensemble), bottom = GradientBoundaryCondition(ensemble_b_bottom))
    ensemble_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ_ensemble), bottom = GradientBoundaryCondition(ensemble_u_bottom))
    coriolis_ensemble = arch_array(architecture, [FPlane(f = get_f₀(observations[j])) for i = 1:Nx, j = 1:Ny])

    ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
        tracers = (:b, :e),
        buoyancy = BuoyancyTracer(),
        boundary_conditions = (; u = ensemble_u_bcs, b = ensemble_b_bcs),
        coriolis = coriolis_ensemble,
        closure = closure_ensemble)

    return ensemble_model
end