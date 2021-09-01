
#
# Single column functionality
#

function initialize_forward_run!(model, data, params, time_index)
    set!(model, params)
    set!(model, data, time_index)
    model.clock.time = data.t[time_index]
    model.clock.iteration = 0
    return nothing
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

#
# Alternative set! implementation
#

function set!(model::EnsembleModel, data_batch::BatchTruthData, time_index)

    # Set the model fields column by column.
    # There's probably a better way to do this.
    for fieldname in [:b, :u, :v, :e]

        for i = 1:ensemble_size(model)
            for j = 1:batch_size(model)
                new_field = getproperty(data_batch[j], fieldname)[time_index]
                old_field = getproperty(model, fieldname)
                @view(old_field[i,j,:]) .= new_field
            end
        end

    end
end

#
# Interpolation functionality that allows multi-dimensional grids but doesn't preserve field budgets
#

"""
    interpolate(field::AbstractField, x, y, z, LX, LY, LZ)

Extends the `interpolate` function from Oceananigans.
Interpolates `field` to the physical point `(x, y, z)` given
field location (LX, LY, LZ) using trilinear interpolation.
"""
function interpolate(field::AbstractField, x, y, z, LX, LY, LZ)

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
    set!(c1::AbstractField, c2::AbstractField)

Set `c1` data to `c2` data interpolated to `c1`'s grid.
"""
function set!(c1::AbstractField, c2::AbstractField)

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


