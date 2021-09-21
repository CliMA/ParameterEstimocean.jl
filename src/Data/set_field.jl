using Oceananigans.Architectures: arch_array, architecture

Lz(c::AbstractDataField) = c.grid.Lz
Nz(c::AbstractDataField) = c.grid.Nz
Δz(g::RegularRectilinearGrid, i::Int) = g.Δz

function integrate_range(c, cgrid, i₁::Int, i₂::Int)
    total = 0
    for i = i₁:i₂
        @inbounds total += c[i] * Δz(cgrid, i)
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
        total = integrate_range(c, c.grid, i₁+1, i₂-1)

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

function integral(c, cgrid, z₋, z₊=0)

    @assert z₊ > cgrid.zF[1] "Integration region lies outside the domain."
    @assert z₊ > z₋ "Invalid integration range: upper limit greater than lower limit."

    # Find region bounded by the face ≤ z₊ and the face ≤ z₁
    i₁ = searchsortedfirst(cgrid.zF, z₋) - 1
    i₂ = searchsortedfirst(cgrid.zF, z₊) - 1

    if i₂ ≠ i₁
        # Calculate interior integral, recalling that the
        # top interior cell has index i₂-2.
        total = integrate_range(c, cgrid, i₁+1, i₂-1)

        # Add contribution to integral from fractional bottom part,
        # if that region is a part of the grid.
        if i₁ > 0
            total += c[i₁] * (cgrid.zF[i₁+1] - z₋)
        end

        # Add contribution to integral from fractional top part
        total += c[i₂] * (z₊ - cgrid.zF[i₂])
    else
        total = c[i₁] * (z₊ - z₋)
    end

    return total
end

# Set interior of field `c` to values of `data`
function set!(c::AbstractField, data::AbstractArray)

    arch = architecture(c)

    # Reshape `data` to the size of `c`'s interior
    d = arch_array(arch, reshape(data, size(c)))

    # Sets the interior of field `c` to values of `data`
    c .= d

end

horizontal_size(grid) = (grid.Nx, grid.Ny)
extent(grid) = (grid.Lx, grid.Ly, grid.Lz)

# Set two fields to one another... some shenanigans
#
_set_similar_fields!(c::AbstractField{Ac, G}, d::AbstractField{Ad, G}) where {Ac, Ad, G} = 
    c.data .= convert(typeof(c.data), d.data)

function interp_and_set!(c1::AbstractField{A1, G1}, c2::AbstractField{A2, G2}) where {A1, A2, G1, G2}

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

"""
    set!(c::AbstractField{Ac, G}, d::AbstractField{Ad, G}) where {Ac, Ad, G}

Set the data of field `c` to the data of field `d`, adjusted to field `c`'s grid. 

The columns are assumed to be independent and thus the fields must have the same 
horizontal resolution. This implementation does not accommodate 3D grids with 
dependent columns.
"""
function set!(c::AbstractField{Ac, G}, d::AbstractField{Ad, G}) where {Ac, Ad, G}

    s1 = horizontal_size(c.grid)
    s2 = horizontal_size(d.grid)
    @assert s1 == s2 "Field grids have a different number of columns."

    if s1 != (1, 1)
        @assert c.grid isa EnsembleGrid && d.grid isa EnsembleGrid "Field has dependent columns."
    end

    if Lz(c) == Lz(d) && Nz(c) == Nz(d)
        return _set_similar_fields!(c, d)
    else
        return interp_and_set!(c, d)
    end

end

