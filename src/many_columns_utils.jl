
ensemble_size(model) = model.grid.Nx
batch_size(model) = model.grid.Ny

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
const BatchTruthData = Vector{<:TruthData}

function many_columns_interior(td_batch, fieldname, time_index, N_ens)
    batch = @. getindex(getproperty(td, fieldname), time_index) # (N_cases, Nz)
    batch = reshape(batch, (1, size(batch)...)) # (1, N_cases, Nz)
    return cat([batch for i = 1:N_ens]..., dims = 1) # (N_ens, N_cases, Nz)
end

# Imitates set! from models_and_data.jl
function set!(model::Oceananigans.AbstractModel,
              td_batch::BatchTruthData, time_index)

    ensemble(x) = many_columns_interior(td_batch, x, time_index, ensemble_size(model))

    set!(model, b = ensemble(:b), 
                u = ensemble(:u),
                v = ensemble(:v),
                e = ensemble(:e)
        )
end