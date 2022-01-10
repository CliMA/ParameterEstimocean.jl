using JLD2
using Oceananigans.Fields: location, XFaceField, YFaceField, ZFaceField, CenterField
using Oceananigans.Grids: Face, Center

location_guide = Dict(:u => (Face, Center, Center),
    :v => (Center, Face, Center),
    :w => (Center, Center, Face))

field_type = Dict(:u => XFaceField,
    :v => YFaceField,
    :w => ZFaceField,
    :b => CenterField,
    :e => CenterField)

function set_with_regrid!(time_series::FieldTimeSeries, path::String, name::String, old_grid, new_grid)

    file = jldopen(path)
    file_iterations = parse.(Int, keys(file["timeseries/t"]))
    file_times = [file["timeseries/t/$i"] for i in file_iterations]
    close(file)

    for (n, time) in enumerate(time_series.times)
        file_index = findfirst(t -> t ≈ time, file_times)
        file_iter = file_iterations[file_index]
    
        field_n = Field(location(time_series), path, name, file_iter,
            boundary_conditions = time_series.boundary_conditions,
            grid = old_grid)
        
        new_field = field_type[Symbol(name)](new_grid)
        regrid!(new_field, field_n)
    
        set!(time_series[n], new_field)
    end

    return nothing
end

# If the location isn't correct in `observations`, `interior` won't be computed correctly
# in InverseProblems.transpose_model_output
function infer_location(field_name)
    if field_name in keys(location_guide)
        return location_guide[field_name]
    else
        return (Center, Center, Center)
    end
end

function legacy_data_field_time_serieses(path, field_names, times; Nz = nothing)

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

    Nz = Nz === nothing ? old_Nz : Nz

    # assumed_location = (Nothing, Nothing, Center)

    boundary_conditions = (; u = u_bcs, b = b_bcs)

    function generate_field_time_series(name)
    
        LX, LY, LZ = infer_location(name)
    
        new_grid = RectilinearGrid(size = Nz, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
    
        time_series = FieldTimeSeries{LX,LY,LZ}(CPU(), new_grid, times, boundary_conditions)
    
        set_with_regrid!(time_series, path, String(name), old_grid, new_grid)
    
        return time_series
    end

    # field_time_serieses = NamedTuple(name => FieldTimeSeries(path, string(name);
    #     times = times,
    #     location = infer_location(name),
    #     grid = grid,
    #     boundary_conditions = (; u = u_bcs, b = b_bcs))
    #                                  for name in field_names)

    field_time_serieses = NamedTuple(name => generate_field_time_series(name) for name in field_names)

    return field_time_serieses
end