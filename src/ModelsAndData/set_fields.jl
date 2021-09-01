
include("set_fields_utils.jl")

# Set interior of field `c` to values of `data`
function set!(c::AbstractField, data::AbstractArray)

    # Reshape `data` to the size of `c`'s interior
    d = reshape(data, size(c))

    # Sets the interior of field `c` to values of `data`
    c .= d

end

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
`set!(ϕ::AbstractField, data)` function exists.

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

function get_interior(data, field_name, time_index)

    field = getproperty(data, field_name)

    # If `time_index` is beyond the range recorded in the simulation output, 
    # then the data for this time step will be ignored down the line, so return zeros
    ans = time_index > length(data) ? zeros(size(interior(field[1]))) :
                                    interior(field[time_index])

    return ans
end

function column_ensemble_interior(data_batch::BatchTruthData, field_name, time_indices::Vector, N_ens)
    batch = @. get_interior(data_batch, field_name, time_indices)
    batch = cat(batch..., dims = 2) # (N_cases, Nz)
    return cat([batch for i = 1:N_ens]..., dims = 1) # (N_ens, N_cases, Nz)
end

"""
    set!(model::EnsembleModel,
         data_batch::BatchTruthData, time_index)

Set columns of each field in `model` to the model profile columns in `data_batch`, 
where every field column in `model` that corresponds to the ith `TruthData` object in `data_batch`
is set to the field column in `data_batch[i]` at time index `time_indices[i]`.
"""
function set!(model::EnsembleModel,
              data_batch::BatchTruthData, time_index::Vector)

    ensemble(x) = column_ensemble_interior(data_batch, x, time_index, model.grid.Nx)

    set!(model, b = ensemble(:b), 
                u = ensemble(:u),
                v = ensemble(:v),
                e = ensemble(:e)
        )
end

set!(model::EnsembleModel, data_batch::BatchTruthData, time_index) = set!(model, data_batch, [time_index for i in data_batch])
