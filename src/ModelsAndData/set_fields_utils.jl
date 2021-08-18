height(c::AbstractDataField) = c.grid.Lz
length(c::AbstractDataField) = c.grid.Nz
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
