
height(c::AbstractDataField) = c.grid.Lz
length(c::AbstractDataField) = c.grid.Nz
Δz(g::RegularRectilinearGrid, i::Int) = g.Δz

function integrate_range(c::AbstractDataField, i₁::Int, i₂::Int)
    total = 0
    for i = i₁:i₂
        @inbounds total += c[i] * Δz(c.grid, i)
    end
    return total
end

function integral(c::AbstractDataField, z₋, z₊=0)

    @assert z₊ > c.grid.zF[1] "Integration region lies outside the domain."
    @assert z₊ > z₋ "Invalid integration range: upper limit greater than lower limit."

    # Find region bounded by the face ≤ z₊ and the face ≤ z₁
    i₁ = searchsortedfirst(c.grid.zF, z₋) - 1
    i₂ = searchsortedfirst(c.grid.zF, z₊) - 1

    if i₂ ≠ i₁
        # Calculate interior integral, recalling that the
        # top interior cell has index i₂-2.
        total = integrate_range(c, i₁+1, i₂-1)

        # Add contribution to integral from fractional bottom part,
        # if that region is a part of the grid.
        if i₁ > 0
            total += c[i₁] * (c.grid.zF[i₁+1] - z₋)
        end

        # Add contribution to integral from fractional top part
        total += c[i₂] * (z₊ - c.grid.zF[i₂])
    else
        total = c[i₁] * (z₊ - z₋)
    end

    return total
end

# Set to an array
function set!(c::AbstractDataField, data::AbstractArray)
    for i in eachindex(data)
        @inbounds c[i] = data[i]
    end
    return nothing
end

# Set two fields to one another... some shenanigans
#
_set_similar_fields!(c::AbstractDataField{Ac, G}, d::AbstractDataField{Ad, G}) where {Ac, Ad, G} = 
    c.data .= convert(typeof(c.data), d.data)

function interp_and_set!(c1::AbstractDataField{A1, G1}, c2::AbstractDataField{A2, G2}) where {A1, A2, G1, G2}
    @assert height(c1) == height(c2) "Physical domains differ between the two fields."
    for i in eachindex(c1.data)
        @inbounds c1[i] = integral(c2, c1.grid.zF[i], c1.grid.zF[i+1]) / Δz(c1.grid, i)
    end
    return nothing
end

## This implementation does not accommodate 3D grids that are not columns
function set!(c::AbstractDataField{Ac, G}, d::AbstractDataField{Ad, G}) where {Ac, Ad, G}
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

end

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

