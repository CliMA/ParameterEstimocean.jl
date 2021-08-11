# https://github.com/CliMA/Oceananigans.jl/blob/master/validation/vertical_mixing_closures/gpu_tkevd_ensemble.jl

function ParameterizedModel(td_batch::Vector{<:TruthData}, Δt; N_ens = 50, kwargs...)

    grid = td_batch[1].grid
    closure = [CATKEVerticalDiffusivity(Float64; warning=false, kwargs...) for i=1:N_ens, j=1:length(td_batch)]
    # coriolis = [td.constants[:f] for i=1:N_ens, td in td_batch]
    coriolis = td_batch[1].constants[:f]

    ensemble(f) = [f(td) for i = 1:N_ens, td in td_batch]

    Qᵇ = ensemble(td -> td.boundary_conditions.Qᶿ * td.constants[:αg])
    Qᵘ = ensemble(td -> td.boundary_conditions.Qᵘ)
    Qᵛ = ensemble(td -> td.boundary_conditions.Qᵛ)
    dbdz_bottom = ensemble(td -> td.boundary_conditions.dθdz_bottom * td.constants[:αg])
    dudz_bottom = ensemble(td -> td.boundary_conditions.dudz_bottom)

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ), bottom = GradientBoundaryCondition(dudz_bottom))
    v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(dbdz_bottom))

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                         tracers = (:b, :e),
                                         buoyancy = BuoyancyTracer(), # SeawaterBuoyancy(eltype(grid))
                                         coriolis = FPlane(f=coriolis),
                                         boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                         closure = closure)

    return OceanTurbulenceParameterEstimation.ParameterizedModel(model, Δt)
end