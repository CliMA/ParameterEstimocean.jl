using Oceananigans: CenterField

# Evaluate loss function AFTER the forward map rather than during
mutable struct ParameterizedModelTimeSeries{UU, VV, BΘ, EE}
                     u :: UU
                     v :: VV
                     b :: BΘ
                     e :: EE
end

function model_time_series(parameters, model, data)

    start = data.targets[1]
    Nt = length(data.t) - start + 1

    initialize_forward_run!(model, data, parameters, start)

    grid = model.grid

    output = ParameterizedModelTimeSeries([XFaceField(grid) for i = 1:Nt],
                             [YFaceField(grid) for i = 1:Nt],
                             [CenterField(grid) for i = 1:Nt],
                             [CenterField(grid) for i = 1:Nt])

    u = getproperty(model, :u)
    v = getproperty(model, :v)
    b = getproperty(model, :b)
    e = getproperty(model, :e)

    for i in 1:Nt
        run_until!(model, data.t[i + start - 1])

        i < 3 && @info "$i,: $([interior(model.b)...])"

        u_snapshot = output.u[i].data
        v_snapshot = output.v[i].data
        b_snapshot = output.b[i].data
        e_snapshot = output.e[i].data

        u_snapshot .= u.data
        v_snapshot .= v.data
        b_snapshot .= b.data
        e_snapshot .= e.data
    end

    return output
end

# Evaluate loss function AFTER the forward map rather than during
function analyze_weighted_profile_discrepancy(loss, forward_map_output::ParameterizedModelTimeSeries, data, target)
    total_discrepancy = zero(eltype(data.grid))
    field_names = Tuple(loss.fields)

    # target = loss.targets[index]
    for (field_index, field_name) in enumerate(field_names)
        model_field = getproperty(forward_map_output, field_name)[target]
        data_field = getproperty(data, field_name)[target]

        # Calculate the per-field profile-based disrepancy
        field_discrepancy = analyze_profile_discrepancy(loss.profile, model_field, data_field)
        if target == 1
            println(field_discrepancy)
        end

        # Accumulate weighted profile-based disrepancies in the total discrepancyor
        total_discrepancy += get_weight(loss.weights, field_index) * field_discrepancy # accumulate discrepancyor
    end

    return nan2inf(total_discrepancy)
end

# Evaluate loss function AFTER the forward map rather than during
function (loss::LossFunction)(forward_map_output::ParameterizedModelTimeSeries, data)
    @inbounds loss.time_series.data[1] = analyze_weighted_profile_discrepancy(loss, forward_map_output, data, loss.targets[1])

    # Calculate a loss function time-series
    for (i, target) in enumerate(loss.targets)
        @inbounds loss.time_series.data[i] =
            analyze_weighted_profile_discrepancy(loss, forward_map_output, data, target)
    end

    return loss.time_series.analysis(loss.time_series.data, loss.time_series.time)
end
