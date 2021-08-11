"""
    function simple_tke_model(constants;
                              grid, Qᵇ, Qᵘ, Qᵛ,
                              dbdz_bottom, dudz_bottom,
             diffusivity_scaling = RiDependentDiffusivityScaling(),
           dissipation_parameter = 2.91,
         mixing_length_parameter = 1.16,
                   # surface_model = TKESurfaceFlux(),
             time_discretization = VerticallyImplicitTimeDiscretization()
                         )

Construct a model with `constants`, grid `grid`,
bottom buoyancy gradient `dbdz`, bottom u-velocity gradient `dudz`,
and forced by

    - buoyancy flux `Qᶿ`
    - x-momentum flux `Qᵘ`
    - y-momentum flux `Qᵛ`.

The keyword arguments `diffusivity_scaling`, `dissipation_parameter`,
`mixing_length_parameter`, `surface_model` and `time_discretization` set
their respective components of the `TKEBasedVerticalDiffusivity` closure
in Oceananigans.
"""
function ParameterizedModel(td::TruthData, Δt; kwargs...)

    αg = td.constants[:αg]
    grid = td.grid
    Qᵇ = td.boundary_conditions.Qᶿ * αg
    Qᵘ = td.boundary_conditions.Qᵘ
    Qᵛ = td.boundary_conditions.Qᵛ
    dbdz_bottom = td.boundary_conditions.dθdz_bottom * αg
    dudz_bottom = td.boundary_conditions.dudz_bottom

    closure = TKEBasedVerticalDiffusivity(Float64; kwargs...)

    # u★ = (Qᵘ^2 + Qᵛ^2)^(1/4)
    # w★³ = Qᵇ * grid.Δz
    # Qᵉ = - closure.dissipation_parameter * (closure.surface_model.CᵂwΔ * w★³ + closure.surface_model.Cᵂu★ * u★^3)

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ), bottom = GradientBoundaryCondition(dudz_bottom))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(dbdz_bottom))
    # tke_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵉ))

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                         tracers = (:b, :e),
                                         buoyancy = BuoyancyTracer(), # SeawaterBuoyancy(eltype(grid))
                                         coriolis = FPlane(f=td.constants[:f]),
                                         # boundary_conditions = (b=b_bcs, e=tke_bcs, u=u_bcs, v=v_bcs),
                                         boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                         closure = closure)

    return OceanTurbulenceParameterEstimation.ParameterizedModel(model, Δt)
end