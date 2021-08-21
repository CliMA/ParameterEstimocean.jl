
include("set_fields_utils.jl")

# Set to an array
function set!(c::AbstractDataField, data::AbstractArray)

    for i in eachindex(data)
        c[i] = data[i]
    end
    return nothing
end

import Base.size
extent(grid::Oceananigans.Grids.AbstractGrid) = (grid.Lx, grid.Ly, grid.Lz)
# size(grid::Oceananigans.Grids.AbstractGrid) = (grid.Nx, grid.Ny, grid.Nz)
horizontal_size(grid::Oceananigans.Grids.AbstractGrid) = (grid.Nx, grid.Ny)

# Set two fields to one another... some shenanigans
#
_set_similar_fields!(c::AbstractDataField{Ac, G}, d::AbstractDataField{Ad, G}) where {Ac, Ad, G} = 
    c.data .= convert(typeof(c.data), d.data)

function interp_and_set!(c1::AbstractDataField{A1, G1}, c2::AbstractDataField{A2, G2}) where {A1, A2, G1, G2}

    grid1 = c1.grid
    grid2 = c2.grid

    @assert extent(grid1) == extent(grid2) "Physical domains differ between the two fields."

    for j in 1:grid1.Ny, i in 1:grid1.Nx
        for k in 1:grid1.Nz
            @inbounds c1[i,j,k] = integral(c2[i,j,:], grid2, grid1.zF[k], grid1.zF[k+1]) / Δz(grid1, i)
        end
    end

    return nothing
end

topology(grid::RegularRectilinearGrid) = Tuple(typeof(grid).parameters[2:4])

independent_columns(grid::RegularRectilinearGrid) = topology(grid) == (Flat, Flat, Bounded)

"""
    set!(c::AbstractDataField{Ac, G}, d::AbstractDataField{Ad, G}) where {Ac, Ad, G}

Set the data of field `c` to the data of field `d`, adjusted to field `c`'s grid. 

The columns are assumed to be independent and thus the fields must have the same 
horizontal resolution. This implementation does not accommodate 3D grids with 
dependent columns.
"""
function set!(c::AbstractDataField{Ac, G}, d::AbstractDataField{Ad, G}) where {Ac, Ad, G}

    s1 = horizontal_size(c.grid)
    s2 = horizontal_size(d.grid)
    @assert s1 == s2 "Field grids have a different number of columns."

    if s1 != (1, 1)
        @assert independent_columns(c.grid) && independent_columns(d.grid) "Field has dependent columns."
    end

    if height(c) == height(d) && length(c) == length(d)
        return _set_similar_fields!(c, d)
    else
        return interp_and_set!(c, d)
    end

end

"""
set!(model; kwargs...)

Set velocity and tracer fields of `model` given data defined on `grid`.
The keyword arguments `kwargs...` take the form `name=data`, where `name` refers
to one of the fields of `model.velocities` or `model.tracers`, and the `data`
may be an array, a function with arguments `(x, y, z)`, or any data type for which a
`set!(ϕ::AbstractDataField, data)` function exists.

Equivalent to `Oceananigans.Models.HydrostaticFreeSurfaceModels.set!` but where the
data in `kwargs...` exist on an arbitrary-resolution grid `grid`
and may need to be interpolated to the model grid `model.grid`.
"""
function set!(model; kwargs...)

    for (fldname, value) in kwargs

        # Get the field `ϕ` corresponding to `fldname`.
        if fldname ∈ propertynames(model.velocities)
            ϕ = getproperty(model.velocities, fldname)
        elseif fldname ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, fldname)
        elseif fldname ∈ propertynames(model.free_surface)
            ϕ = getproperty(model.free_surface, fldname)
        else
            throw(ArgumentError("name $fldname not found in model.velocities or model.tracers."))
        end

        set!(ϕ, value)
    end

    return nothing
end

function get_interior(td, field_name, time_index)

    field = getproperty(td, field_name)

    # If `time_index` is beyond the range recorded in the simulation output, 
    # then the data for this time step will be ignored down the line, so return zeros
    # ans = time_index > length(td) ? zeros(size(interior(field[1]))) :
    #                                 interior(field[time_index])
    ans = time_index < length(td) ? zeros(size(interior(field[1]))) :
                                    interior(field[time_index])

    return ans
end

function column_ensemble_interior(td_batch::BatchTruthData, field_name, time_indices::Vector, N_ens)
    batch = @. get_interior(td_batch, field_name, time_indices)
    batch = cat(batch..., dims = 2) # (N_cases, Nz)
    return cat([batch for i = 1:N_ens]..., dims = 1) # (N_ens, N_cases, Nz)
end

"""
    set!(model::Oceananigans.AbstractModel,
                data::TruthData, i)

Set the fields of `model` to the fields of `data` at time step `i`,
and adjust the model clock time accordingly.
"""
function set!(model, data::TruthData, i)

    set!(model, b = data.b[i],
                u = data.u[i],
                v = data.v[i],
                e = data.e[i]
        )

    model.clock.time = data.t[i]

    return nothing
end

"""
set!(model::AbstractModel,
              td_batch::BatchTruthData, time_index)


"""
function set!(model::ParameterizedModel,
              td_batch::BatchTruthData, time_index::Vector)

    ensemble(x) = column_ensemble_interior(td_batch, x, time_index, model.grid.Nx)

    set!(model, b = ensemble(:b), 
                u = ensemble(:u),
                v = ensemble(:v),
                e = ensemble(:e)
        )
end

set!(model::Oceananigans.AbstractModel, td_batch::BatchTruthData, time_index) = set!(model, td_batch, [time_index for i in td_batch])

# function set!(model::ParameterizedModel, td_batch::BatchTruthData, time_index)

#     # Set the model fields column by column.
#     # There's probably a better way to do this.
#     for fieldname in [:b, :u, :v, :e]

#         for i = 1:ensemble_size(model)
#             for j = 1:batch_size(model)
#                 new_field = getproperty(td_batch[j], fieldname)[time_index]
#                 old_field = getproperty(model, fieldname)
#                 @view(old_field[i,j,:]) .= new_field
#             end
#         end

#     end
# end


## BELOW: interpolation functionality that allows multi-dimensional grids but doesn't preserve field budgets

#=
"""
    interpolate(field::AbstractDataField, x, y, z, LX, LY, LZ)

Extends the `interpolate` function from Oceananigans.
Interpolates `field` to the physical point `(x, y, z)` given
field location (LX, LY, LZ) using trilinear interpolation.
"""
function interpolate(field::AbstractDataField, x, y, z, LX, LY, LZ)

    grid = field.grid

    i = isnothing(LX) ? 0.0 : Oceananigans.Fields.fractional_x_index(x, LX(), grid)
    j = isnothing(LY) ? 0.0 : Oceananigans.Fields.fractional_y_index(y, LY(), grid)
    k = isnothing(LZ) ? 0.0 : Oceananigans.Fields.fractional_z_index(z, LZ(), grid)

    # Convert fractional indices to unit cell coordinates 0 <= (ξ, η, ζ) <=1
    # and integer indices (with 0-based indexing).
    ξ, i = modf(i)
    η, j = modf(j)
    ζ, k = modf(k)

    # Convert indices to proper integers and shift to 1-based indexing.
    return Oceananigans.Fields._interpolate(field, ξ, η, ζ, Int(i+1), Int(j+1), Int(k+1))
end

"""
    get_loc_from_last_index(last_index, N, H)

Deduce location `loc` ∈ {Center, Face} from last index along
a dimension with `N` grid points and halo size `H`.
"""
function get_loc_from_last_index(last_index, N, H)
    loc = last_index == 1         ? nothing :
          last_index == N + H     ? Center :
          last_index == N + H + 1 ? Face :
          throw(ErrorException)
    return loc
end

function get_loc(data, grid)

    x_indices, y_indices, z_indices = axes(data)
    LX = get_loc_from_last_index(x_indices[end], grid.Nx, grid.Hx)
    LY = get_loc_from_last_index(y_indices[end], grid.Ny, grid.Hy)
    LZ = get_loc_from_last_index(z_indices[end], grid.Nz, grid.Hz)

    return (LX, LY, LZ)
end

"""
    set!(c1::AbstractDataField, c2::AbstractDataField)

Set `c1` data to `c2` data interpolated to `c1`'s grid.
"""
function set!(c1::AbstractDataField, c2::AbstractDataField)

    # if size(c1.data) == size(c2.data)
    if c1.grid == c2.grid
        c1.data .= c2.data
    else
        # all coordinates in `ϕ` array
        g1 = c1.grid

        # Deduce location ∈ {Center, Face} from last index along
        # each dimension of the data for field `c1`
        LX, LY, LZ = get_loc(c1, g1) ## Reverse engineeering not a great solution! Alternatives?

        # Physical coordinates corresponding to each index in `c1` data along each dimension
        x_coordinates = LX == Face ? g1.xF : g1.xC
        y_coordinates = LY == Face ? g1.yF : g1.yC
        z_coordinates = LZ == Face ? g1.zF : g1.zC

        # set `c1` to the values of `c2` interpolated to `c1`'s grid.
        for (x, y, z) in axes(c1)
            c1[x, y, z] = interpolate(c2, x_coordinates[x],
                                          y_coordinates[y],
                                          z_coordinates[z],
                                          LX, LY, LZ)
        end
    end

end
=#

