
const EnsembleModel = HydrostaticFreeSurfaceModel{TS, E, A, S, <:EnsembleGrid, T, V, B, R, F, P, U, C, Φ, K, AF} where {TS, E, A, S, T, V, B, R, F, P, U, C, Φ, K, AF}

ensemble_size(model::EnsembleModel) = model.grid.Nx
batch_size(model::EnsembleModel) = model.grid.Ny

"""
    set!(model::EnsembleModel,
         data_batch::TruthDataBatch, time_index)

Set columns of each field in `model` to the model profile columns in `data_batch`, 
where every field column in `model` that corresponds to the ith `TruthData` object in `data_batch`
is set to the field column in `data_batch[i]` at time index `time_indices[i]`.
"""
function set!(model::EnsembleModel,
              data_batch::TruthDataBatch, time_indices::Vector)

    ensemble(x) = column_ensemble_interior(data_batch, x, time_indices, model.grid.Nx)

    set!(model, b = ensemble(:b), 
                u = ensemble(:u),
                v = ensemble(:v),
                e = ensemble(:e)
        )
end

set!(model::EnsembleModel, data_batch::TruthDataBatch, time_index) = set!(model, data_batch, [time_index for i in data_batch])

"""
    initialize_forward_run!(model, data_batch::TruthDataBatch, params, time_indices::Vector)

Set columns of each field in `model` to the corresponding profile columns in `data_batch`, 
where every field column in `model` that corresponds to the ith `TruthData` object in `data_batch`
is set to the field column in `data_batch[i]` at `time_indices[i]`.
"""
function initialize_forward_run!(model::EnsembleModel, data_batch::TruthDataBatch, params, time_indices::Vector)
    set!(model, params)
    set!(model, data_batch, time_indices)
    model.clock.time = 0.0
    model.clock.iteration = 0
    return nothing
end
