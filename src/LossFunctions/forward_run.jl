using Oceananigans.Utils: prettytime

struct ForwardMap{V,D,P}
    model_time_series::V
    data_batch::D
    parameters::P
end

mutable struct ModelTimeSeries{U, V, B, E}
    u :: U
    v :: V
    b :: B
    e :: E
end

ModelTimeSeries(grid, targets) = ModelTimeSeries([XFaceField(grid) for i = targets],
                                                 [YFaceField(grid) for i = targets],
                                                 [CenterField(grid) for i = targets],
                                                 [CenterField(grid) for i = targets])

function model_time_series(simulation, data_batch, parameters)

    model = simulation.model

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

    start_time = time_ns()

    for save_index in 1:max_simulation_length
        simulation.stop_time = t[save_index]
        run!(simulation)

        u = model.velocities.u
        v = model.velocities.v
        b = model.tracers.b
        e = model.tracers.e

        capture_model_state!(outputs, save_index, data_batch, u, v, b, e)
    end

    end_time = time_ns()
    elapsed_time = (end_time - start_time) * 1e-9

    @info "The forward run took $(prettytime(elapsed_time))"

    return ForwardMap(outputs, data_batch, parameters)
end

function capture_model_state!(outputs, save_index, data_batch, u, v, b, e)
    for (data_index, data) in enumerate(data_batch)
        output = outputs[data_index]
        if save_index <= length(data.targets)
            u_snapshot = parent(output.u[save_index])
            v_snapshot = parent(output.v[save_index])
            b_snapshot = parent(output.b[save_index])
            e_snapshot = parent(output.e[save_index])

            copyto!(u_snapshot, view(parent(u), 1, data_index, :))
            copyto!(v_snapshot, view(parent(v), 1, data_index, :))
            copyto!(b_snapshot, view(parent(b), 1, data_index, :))
            copyto!(e_snapshot, view(parent(e), 1, data_index, :))
        end
    end

    return nothing
end
