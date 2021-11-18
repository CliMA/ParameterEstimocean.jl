using Oceananigans
using Oceananigans.Architectures: arch_array
using Oceananigans: AbstractModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel

# function get_parameter(filename, group, parameter_name, default = nothing)
#     parameter = default

#     jldopen(filename) do file
#         if parameter_name ∈ keys(file["$group"])
#             parameter = file["$group/$parameter_name"]
#         end
#     end

#     return parameter
# end

function get_parameter(observation, parameter_path)
    file = jldopen(observation.path)
    parameter = file[parameter_path]
    close(file)
    return parameter
end

"""
    OneDimensionalEnsembleModel(observations::OneDimensionalTimeSeriesBatch; architecture = CPU(), ensemble_size = 50, kwargs...)

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
`OneDimensionalTimeSeries` objects in `observations`.
"""
# function OneDimensionalEnsembleModel(observations;
#     architecture = CPU(),
#     ensemble_size = 50,
#     closure = CATKEVerticalDiffusivity(Float64; warning = false)
# )

#     #
#     # Harvest observation file metadata
#     #

#     datapath = observations.path

#     α = get_parameter(datapath, "buoyancy", "equation_of_state/α", 2e-4)
#     g = get_parameter(datapath, "buoyancy", "gravitational_acceleration", 9.81)
#     αg = α * g

#     f = get_parameter(datapath, "coriolis", "f", 0.0)

#     # Surface fluxes
#     Qᵘ = get_parameter(datapath, "parameters", "boundary_condition_u_top", 0.0)
#     Qᵛ = get_parameter(datapath, "parameters", "boundary_conditions_v_top", 0.0)
#     Qᶿ = get_parameter(datapath, "parameters", "boundary_condition_θ_top", 0.0)
#     Qᵇ = Qᶿ * αg

#     # Bottom gradients
#     dudz_bottom = get_parameter(datapath, "parameters", "boundary_condition_u_bottom", 0.0)
#     dθdz_bottom = get_parameter(datapath, "parameters", "boundary_condition_θ_bottom", 0.0)
#     dbdz_bottom = dθdz_bottom * αg

#     #
#     # Build model using metadata
#     #

#     grid = OneDimensionalEnsembleGrid(observations.grid; size = (ensemble_size, length(observations), data_grid.Nz))

#     closure = [closure for i = 1:ensemble_size, j = 1:length(observations)]

#     coriolis = [FPlane(f = td.constants[:f]) for i = 1:ensemble_size, td in observations]
#     coriolis = arch_array(architecture, coriolis)

#     bc_matrix(x) = [x for i = 1:ensemble_size, d in observations]
#     Qᵇ = bc_matrix(Qᵇ)
#     Qᵘ = bc_matrix(Qᵘ)
#     Qᵛ = bc_matrix(Qᵛ)
#     dbdz_bottom = bc_matrix(dbdz_bottom)
#     dudz_bottom = bc_matrix(dudz_bottom)

#     u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ),
#         bottom = GradientBoundaryCondition(dudz_bottom))
#     v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
#     b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ),
#         bottom = GradientBoundaryCondition(dbdz_bottom))

#     model = HydrostaticFreeSurfaceModel(architecture = architecture,
#         grid = grid,
#         tracers = (:b, :e),
#         buoyancy = BuoyancyTracer(),
#         coriolis = coriolis,
#         boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs),
#         closure = closure)

#     return model
# end

#####
##### Set up ensemble model
#####

# vectorize(observation) = [observation]
# vectorize(observations::Vector) = observations

function OneDimensionalEnsembleModel(observations;
    architecture = CPU(),
    ensemble_size = 50,
    closure = CATKEVerticalDiffusivity(Float64; warning = false)
)
    observations = vectorize(observations)
    @show observations

    Lz = get_parameter(observations[1].path, "grid/Lz")
    g = get_parameter(observations[1].path, "buoyancy", "gravitational_acceleration")
    α = get_parameter(observations[1].path, "buoyancy", "equation_of_state/α")

    Nx = ensemble_size
    Ny = length(observations)
    column_ensemble_size = ColumnEnsembleSize(Nz = Nz, ensemble = (Nx, Ny), Hz = 1)

    ensemble_grid = RegularRectilinearGrid(size = column_ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
    closure_ensemble = [closure for i = 1:Nx, j = 1:Ny]

    get_Qᵇ(observation) = get_parameter(observation, "parameters/boundary_condition_θ_top") * α * g
    get_Qᵘ(observation) = get_parameter(datapath, "parameters/boundary_condition_u_top")
    get_f₀(observation) = get_parameter(observation.path, "coriolis/f", default)

    Qᵇ_ensemble = [get_Qᵇ(observations[j]) for i = 1:Nx, j = 1:Ny]
    Qᵘ_ensemble = [get_Qᵘ(observations[j]) for i = 1:Nx, j = 1:Ny]

    # Convert to CuArray if necessary
    closure = arch_array(architecture, closure)
    Qᵇ_ensemble = arch_array(architecture, Qᵇ_ensemble)
    Qᵘ_ensemble = arch_array(architecture, Qᵘ_ensemble)
    Qᵛ_ensemble = arch_array(architecture, Qᵛ_ensemble)

    ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ_ensemble), bottom = GradientBoundaryCondition(N²))
    ensemble_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ_ensemble))
    coriolis_ensemble = [FPlane(f = get_f₀(observations[j])) for i = 1:Nx, j = 1:Ny]

    ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
        tracers = (:b, :e),
        buoyancy = BuoyancyTracer(),
        boundary_conditions = (; u = ensemble_u_bcs, b = ensemble_b_bcs),
        coriolis = coriolis_ensemble,
        closure = closure_ensemble)

    return ensemble_model
end