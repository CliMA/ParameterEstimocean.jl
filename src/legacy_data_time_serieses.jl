using JLD2

function legacy_data_field_time_serieses(path, field_names, times)

    # Build a grid, assuming it's a 1D RectilinearGrid
    file = jldopen(path)

    Nz = file["grid/Nz"]
    Hz = file["grid/Hz"]
    Lz = file["grid/Lz"]

    close(file)

    grid = RectilinearGrid(size=Nz, halo=Hz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

    assumed_location = (Nothing, Nothing, Center)

    field_time_serieses = NamedTuple(name => FieldTimeSeries(path, string(name);
                                                             times = times,
                                                             location = assumed_location,
                                                             grid = grid,
                                                             boundary_conditions = nothing)
                                     for name in field_names)

    return field_time_serieses
end
