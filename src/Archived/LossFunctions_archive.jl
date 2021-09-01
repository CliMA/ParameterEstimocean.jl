#
# A "master" loss function type
#

abstract type AbstractLossContainer <: Function end

const ALC = AbstractLossContainer

"""
    struct LossFunction{R, F, W, T, P}

A loss function for the analysis of single column models.
"""
struct LossFunction{R, F, W, T, P}
        targets :: R
         fields :: F
        weights :: W # field weights
    time_series :: T
        profile :: P
end

function LossFunction(model, data; 
                          weights = nothing,
                      time_series = TimeSeriesAnalysis(data.t[targets], TimeAverage()),
                          profile = ValueProfileAnalysis(model.grid)
                      )

    return LossFunction(data.targets, data.relevant_fields, weights, time_series, profile)
end

function (loss::LossFunction)(θ, model, data, Δt)
    evaluate!(loss, θ, model, data)
    return loss.time_series.analysis(loss.time_series.data, loss.time_series.time)
end

mutable struct LossContainer{M<:AbstractModel, D<:TruthData, L<:LossFunction, Δ} <: ALC
    model :: M
     data :: D
     loss :: L
       Δt :: Δ
end

(lc::LossContainer)(θ) = lc.loss(θ, lc.model, lc.data, lc.Δt)

#
# Batched loss function
#

mutable struct BatchedLossContainer{B, W, E} <: ALC
      batch :: B
    weights :: W # scenario weights
      error :: E
end

function BatchedLossContainer(batch; weights=[1.0 for b in batch])
    return BatchedLossContainer(batch, weights, zeros(length(batch)))
end

function (bl::BatchedLossContainer)(θ::FreeParameters)
    bl.error .= 0
    @inbounds begin
        Base.Threads.@threads for i = 1:length(bl.batch)
            bl.error[i] = bl.weights[i] * bl.batch[i](θ)
        end
    end
    return sum(bl.error)
end

const BatchLossFunction = Vector{<:LossFunction}


function analyze_weighted_profile_discrepancy(loss, model, data::TruthData, target)
    total_discrepancy = zero(eltype(model.grid))
    field_names = Tuple(loss.fields)

    for (field_index, field_name) in enumerate(field_names)
        model_field = getproperty(model, field_name)
        data_field = getproperty(data, field_name)[target]

        # Calculate the per-field profile-based disrepancy
        field_discrepancy = analyze_profile_discrepancy(loss.profile, model_field, data_field)

        # Accumulate weighted profile-based disrepancies in the total discrepancyor
        total_discrepancy += get_weight(loss.weights, field_index) * field_discrepancy # accumulate discrepancyor
    end

    return nan2inf(total_discrepancy)
end

function evaluate!(loss, parameters, model_plus_Δt, data::TruthData)

    # Initialize
    initialize_forward_run!(model_plus_Δt, data, parameters, loss.targets[1])

    @inbounds loss.time_series.data[1] = analyze_weighted_profile_discrepancy(loss, model_plus_Δt, data, loss.targets[1])

    # Calculate a loss function time-series
    for (i, target) in enumerate(loss.targets)
        run_until!(model_plus_Δt.model, loss.Δt, data.t[target])

        @inbounds loss.time_series.data[i] =
            analyze_weighted_profile_discrepancy(loss, model_plus_Δt, data, target)
    end

    return nothing
end

function init_loss_function(model::AbstractModel, data::TruthData,
                                      relative_weights; analysis = mean)

    grid = model.grid
    profile_analysis = ValueProfileAnalysis(grid, analysis = analysis)
    profile_analysis = on_grid(profile_analysis, grid)
    weights = estimate_weights(profile_analysis, data, relative_weights)

    loss_function = LossFunction(model, data, weights=weights,
                        time_series = TimeSeriesAnalysis(data.t[data.targets], TimeAverage()),
                        profile = profile_analysis)

    return loss_function
end

struct VarianceWeights{F, D, T, V}
       fields :: F
         data :: D
      targets :: T
    variances :: V
end

@inbounds normalize_variance(::Nothing, field, σ) = σ

function VarianceWeights(data; fields, targets=1:length(data), normalizer=nothing)
    variances = (; zip(fields, (zeros(length(targets)) for field in fields))...)

    for (k, field) in enumerate(fields)
        for i in 1:length(targets)
            @inbounds variances[k][i] = normalize_variance(normalizer, field, variance(data, field, i))
        end
    end

    return VarianceWeights(fields, data, targets, variances)
end

function simple_safe_save(savename, variable, name="calibration")

    temppath = savename[1:end-5] * "_temp.jld2"
    newpath = savename

    isfile(newpath) && mv(newpath, temppath, force=true)

    println("Saving to $savename...")
    save(newpath, name, variable)

    isfile(temppath) && rm(temppath)

    return nothing
end


function model_time_series(parameters, model, data, Δt)

    start = data.targets[1]
    Nt = length(data.t) - start + 1

    initialize_forward_run!(model, data, parameters, start)

    grid = model.grid

    output = ModelTimeSeries([XFaceField(grid) for i = 1:Nt],
                             [YFaceField(grid) for i = 1:Nt],
                             [CenterField(grid) for i = 1:Nt],
                             [CenterField(grid) for i = 1:Nt])

    u = getproperty(model, :u)
    v = getproperty(model, :v)
    b = getproperty(model, :b)
    e = getproperty(model, :e)

    for i in 1:Nt
        run_until!(model, Δt, data.t[i + start - 1])

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
function analyze_weighted_profile_discrepancy(loss, forward_map_output::ModelTimeSeries, data, target)
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
function (loss::LossFunction)(forward_map_output::ModelTimeSeries, data)
    @inbounds loss.time_series.data[1] = analyze_weighted_profile_discrepancy(loss, forward_map_output, data, loss.targets[1])

    # Calculate a loss function time-series
    for (i, target) in enumerate(loss.targets)
        @inbounds loss.time_series.data[i] =
            analyze_weighted_profile_discrepancy(loss, forward_map_output, data, target)
    end

    return loss.time_series.analysis(loss.time_series.data, loss.time_series.time)
end
