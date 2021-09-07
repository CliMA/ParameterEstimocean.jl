
mutable struct ModelTimeSeries{UU, VV, BΘ, EE}
    u :: UU
    v :: VV
    b :: BΘ
    e :: EE
end

ModelTimeSeries(grid, targets) = ModelTimeSeries([XFaceField(grid) for i = targets],
                                                 [YFaceField(grid) for i = targets],
                                                 [CenterField(grid) for i = targets],
                                                 [CenterField(grid) for i = targets])

function model_time_series(parameters, model, data_batch, Δt)

    # Sometimes data_batch will have fewer data objects than the model, so pad data_batch with redundant objects
    redundant_data = [data_batch[1] for _ in 1:(batch_size(model) - length(data_batch))]
    full_data_batch = [data_batch; redundant_data]
    
    all_targets = getproperty.(full_data_batch, :targets)
    max_simulation_length = maximum(length.(all_targets))
    starts = getindex.(all_targets, 1)

    initialize_forward_run!(model, full_data_batch, parameters, starts)

    outputs = [ModelTimeSeries(data.grid, data.targets) for data in data_batch]

    u = get_model_field(model, :u)
    v = get_model_field(model, :v)
    b = get_model_field(model, :b)
    e = get_model_field(model, :e)

    # this should be improved
    all_lengths = length.(getproperty.(data_batch, :t))
    longest_sim = data_batch[argmax(all_lengths)]
    t = longest_sim.t # times starting from zero

    simulation = Simulation(model; Δt=Δt, stop_time = 0.0)
    # pop!(simulation.diagnostics, :nan_checker)

    #for i in 1:max_simulation_length
    for i in 1:3
        
        simulation.stop_time = t[i]
        run!(simulation)

        for (dataindex, data, start, output) in zip(eachindex(data_batch), data_batch, starts, outputs)

            if i <= length(data.targets)
                u_snapshot = output.u[i].data
                v_snapshot = output.v[i].data
                b_snapshot = output.b[i].data
                e_snapshot = output.e[i].data

                u_snapshot .= u.data[1:1, dataindex:dataindex, :]
                v_snapshot .= v.data[1:1, dataindex:dataindex, :]
                b_snapshot .= b.data[1:1, dataindex:dataindex, :]
                e_snapshot .= e.data[1:1, dataindex:dataindex, :]
            end
        end
    end

    return outputs
end
