using Oceananigans: CenterField

# Evaluate loss function AFTER the forward map rather than during
mutable struct ParameterizedModelTimeSeries{UU, VV, BΘ, EE}
                     u :: UU
                     v :: VV
                     b :: BΘ
                     e :: EE
end

function model_time_series(parameters, loss)

    model_plus_Δt = loss.model
    data = loss.data

    # Nt = length(data.t)

    # start = 1
    start = loss.targets[1]
    Nt = length(data.t) - start + 1

    initialize_forward_run!(model_plus_Δt, data, parameters, start)

    grid = model_plus_Δt.grid

    output = ParameterizedModelTimeSeries([CenterField(grid) for i = 1:Nt],
                             [CenterField(grid) for i = 1:Nt],
                             [CenterField(grid) for i = 1:Nt],
                             [CenterField(grid) for i = 1:Nt])

    u = getproperty(model_plus_Δt, :u)
    v = getproperty(model_plus_Δt, :v)
    b = getproperty(model_plus_Δt, :b)
    e = getproperty(model_plus_Δt, :e)
    # e = 0

    for i in 1:Nt
        run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data.t[i + start - 1])

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


# function model_time_series(parameters, model_plus_Δt, data)
#
#     Nt = length(data.t)
#
#     targets[1]
#
#     initialize_forward_run!(model_plus_Δt, data, parameters, 1)
#
#     grid = model_plus_Δt.grid
#
#     output = ParameterizedModelTimeSeries([CenterField(grid) for i = 1:Nt],
#                               [CenterField(grid) for i = 1:Nt],
#                               [CenterField(grid) for i = 1:Nt],
#                               [CenterField(grid) for i = 1:Nt],
#                               [CenterField(grid) for i = 1:Nt])
#
#     U = model_plus_Δt.solution.U
#     V = model_plus_Δt.solution.V
#     T = model_plus_Δt.solution.T
#     S = model_plus_Δt.solution.S
#     e = model_plus_Δt.solution.e
#
#     for i in 1:Nt
#         run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data.t[i])
#
#         U_snapshot = output.U[i].data
#         V_snapshot = output.V[i].data
#         T_snapshot = output.T[i].data
#         S_snapshot = output.S[i].data
#         e_snapshot = output.e[i].data
#
#         U_snapshot .= U.data
#         V_snapshot .= V.data
#         T_snapshot .= T.data
#         S_snapshot .= S.data
#         e_snapshot .= e.data
#     end
#
#     return output
# end

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
