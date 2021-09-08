
const OneDimensionalEnsembleModel = HydrostaticFreeSurfaceModel{TS, E, A, S, <:OneDimensionalEnsembleGrid, T, V, B, R, F, P, U, C, Φ, K, AF} where {TS, E, A, S, T, V, B, R, F, P, U, C, Φ, K, AF}

ensemble_size(model::OneDimensionalEnsembleModel) = model.grid.Nx
batch_size(model::OneDimensionalEnsembleModel) = model.grid.Ny

"""
    set!(model::OneDimensionalEnsembleModel,
         observations::OneDimensionalTimeSeriesBatch, time_index)

Set columns of each field in `model` to the model profile columns in `observations`, 
where every field column in `model` that corresponds to the ith `OneDimensionalTimeSeries` object in `observations`
is set to the field column in `observations[i]` at time index `time_indices[i]`.
"""
function set!(model::OneDimensionalEnsembleModel,
              observations::OneDimensionalTimeSeriesBatch, time_indices::Vector)

    ensemble(x) = column_ensemble_interior(observations, x, time_indices, model.grid.Nx)

    set!(model, b = ensemble(:b), 
                u = ensemble(:u),
                v = ensemble(:v),
                e = ensemble(:e)
        )
end

set!(model::OneDimensionalEnsembleModel, observations::OneDimensionalTimeSeriesBatch, time_index) = set!(model, observations, [time_index for i in observations])

"""
    initialize_forward_run!(model, observations::OneDimensionalTimeSeriesBatch, params, time_indices::Vector)

Set columns of each field in `model` to the corresponding profile columns in `observations`, 
where every field column in `model` that corresponds to the ith `OneDimensionalTimeSeries` object in `observations`
is set to the field column in `observations[i]` at `time_indices[i]`.
"""
function initialize_forward_run!(model::OneDimensionalEnsembleModel, observations::OneDimensionalTimeSeriesBatch, params, time_indices::Vector)
    set!(model, params)
    set!(model, observations, time_indices)
    model.clock.time = 0.0
    model.clock.iteration = 0
    return nothing
end
