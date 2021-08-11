function ParameterizedModel(td::TruthData, Δt; kwargs...)

    αg = td.constants[:αg]
    grid = td.grid
    Qᵇ = td.boundary_conditions.Qᶿ * αg
    Qᵘ = td.boundary_conditions.Qᵘ
    Qᵛ = td.boundary_conditions.Qᵛ
    dbdz_bottom = td.boundary_conditions.dθdz_bottom * αg
    dudz_bottom = td.boundary_conditions.dudz_bottom

    closure = CATKEVerticalDiffusivity(Float64; warning=false, kwargs...)

    # u★ = (Qᵘ^2 + Qᵛ^2)^(1/4)
    # w★³ = Qᵇ * grid.Δz
    # Qᵉ = - closure.dissipation_parameter * (closure.surface_TKE_flux.CᵂwΔ * w★³ + closure.surface_TKE_flux.Cᵂu★ * u★^3)

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