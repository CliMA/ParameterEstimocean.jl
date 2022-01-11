using JLD2
using Oceananigans.Fields: location, XFaceField, YFaceField, ZFaceField, CenterField
using Oceananigans.Grids: Face, Center

location_guide = Dict(:u => (Face, Center, Center),
    :v => (Center, Face, Center),
    :w => (Center, Center, Face))

# If the location isn't correct in `observations`, `interior` won't be computed correctly
# in InverseProblems.transpose_model_output
function infer_location(field_name)
    if field_name in keys(location_guide)
        return location_guide[field_name]
    else
        return (Center, Center, Center)
    end
end

function legacy_data_field_time_serieses(path, field_names, times)

    # Build a grid, assuming it's a 1D RectilinearGrid
    file = jldopen(path)

    old_Nz = file["grid/Nz"]
    Hz = file["grid/Hz"]
    Lz = file["grid/Lz"]

    αg = file["buoyancy/gravitational_acceleration"] * file["buoyancy/equation_of_state/α"]
    Qᵇ = file["parameters/boundary_condition_θ_top"] * αg
    Qᵘ = file["parameters/boundary_condition_u_top"]
    u_bottom = file["parameters/boundary_condition_u_bottom"]
    b_bottom = file["parameters/boundary_condition_θ_bottom"] * αg

    close(file)

    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(b_bottom))
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ), bottom = GradientBoundaryCondition(u_bottom))

    old_grid = RectilinearGrid(size = old_Nz, halo = Hz, z = (-Lz, 0), topology = (Flat, Flat, Bounded))

    boundary_conditions = (; u = u_bcs, b = b_bcs)

    field_time_serieses = NamedTuple(name => FieldTimeSeries(path, string(name);
        times = times,
        location = infer_location(name),
        grid = grid,
        boundary_conditions = boundary_conditions)
                                     for name in field_names)

    field_time_serieses = NamedTuple(name => generate_field_time_series(name) for name in field_names)

    return field_time_serieses
end