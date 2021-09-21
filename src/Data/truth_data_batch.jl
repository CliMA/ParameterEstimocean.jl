

const TruthDataBatch = Vector{<:TruthData}

function get_interior(data, field_name, time_index)

    field = getproperty(data, field_name)

    # If `time_index` is beyond the range recorded in the simulation output, 
    # then the data for this time step will be ignored down the line, so return zeros
    ans = time_index > length(data.t) ? zeros(size(interior(field[1]))) :
                                    interior(field[time_index])

    return ans
end

"""
    column_ensemble_interior(data_batch::TruthDataBatch, field_name, time_indices::Vector, N_ens)

Returns an `N_cases × N_ens` array of the interior of a field `field_name` defined on a 
`ColumnEnsembleGrid` of size `N_cases × N_ens × Nz`, given a list of `TruthData` objects
containing the `N_cases` single-column fields at the corresponding time index in `time_indices`.
"""
function column_ensemble_interior(data_batch::TruthDataBatch, field_name, time_indices::Vector, N_ens)
    batch = @. get_interior(data_batch, field_name, time_indices)
    batch = cat(batch..., dims = 2) # (N_cases, Nz)
    return cat([batch for i = 1:N_ens]..., dims = 1) # (N_ens, N_cases, Nz)
end