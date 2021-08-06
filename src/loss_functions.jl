#
# A "master" loss function type
#

abstract type AbstractLossFunction <: Function end

const ALOSS = AbstractLossFunction

"""
    struct LossFunction{M, D, R, F, W, T, P}

A loss function for the analysis of single column models.
"""
mutable struct LossFunction{M, D, R, F, W, T, P} <: ALOSS
          model :: M
           data :: D
        targets :: R
         fields :: F
        weights :: W # field weights
    time_series :: T
        profile :: P
end

function LossFunction(model, data; fields,
                          targets = 1:length(data.t),
                          weights = nothing,
                      time_series = TimeSeriesAnalysis(data.t[targets], TimeAverage()),
                          profile = ValueProfileAnalysis(model.grid)
                      )

    return LossFunction(model, data, targets, fields, weights, time_series, profile)
end

function (loss::LossFunction)(θ)
    evaluate!(loss, θ, loss.model, loss.data)
    return loss.time_series.analysis(loss.time_series.data, loss.time_series.time)
end

#
# Batched loss function
#

mutable struct BatchedLossFunction{B, W, E} <: ALOSS
      batch :: B
    weights :: W # simulation weights
      error :: E
end

function BatchedLossFunction(batch; weights=[1.0 for b in batch])
    return BatchedLossFunction(batch, weights, zeros(length(batch)))
end

function (bl::BatchedLossFunction)(θ)
    bl.error .= 0
    @inbounds begin
        Base.Threads.@threads for i = 1:length(bl.batch)
            bl.error[i] = bl.weights[i] * bl.batch[i](θ)
        end
    end
    return sum(bl.error)
end

#
# Time analysis
#

struct TimeSeriesAnalysis{T, D, A}
        time :: T
        data :: D
    analysis :: A
end

TimeSeriesAnalysis(time, analysis) = TimeSeriesAnalysis(time, zeros(length(time)), analysis)

struct TimeAverage end

# Use trapz integral to compute time average of data in case times are not evenly spaced
@inline (::TimeAverage)(data, time) = trapz(data, time) / (time[end] - time[1])

#
# Profile analysis
#

"""
    struct ValueProfileAnalysis{D, A}

A type for doing analyses on a discrepancy profile located
at cell centers. Defaults to taking the mean square difference between
the model and data coarse-grained to the model grid.
"""
struct ValueProfileAnalysis{D, A}
    discrepancy :: D
       analysis :: A
end

ValueProfileAnalysis(grid; analysis=mean) = ValueProfileAnalysis(CenterField(grid), analysis)
ValueProfileAnalysis(; analysis=mean) = ValueProfileAnalysis(nothing, analysis)
on_grid(profile::ValueProfileAnalysis, grid) = ValueProfileAnalysis(grid; analysis=profile.analysis)

function calculate_value_discrepancy!(value, model_field, data_field)
    coarse_grained = discrepancy = value.discrepancy
    set!(coarse_grained, data_field)

    for i in eachindex(discrepancy)
        @inbounds discrepancy[i] = (coarse_grained[i] - model_field[i])^2
    end
    return nothing
end

"""
    analyze_profile_discrepancy(value, model_field, data_field)

Calculates the discrepancy between model and data field values, and returns an
analysis of the discrepancy profile.
"""
function analyze_profile_discrepancy(value, model_field, data_field)
    calculate_value_discrepancy!(value, model_field, data_field)
    return value.analysis(value.discrepancy)
end

"""
    struct GradientProfileAnalysis{D, A}

A type for combining discreprancy between the fields and field gradients.
Defaults to taking the mean square difference between
the model and data coarse-grained to the model grid.
"""
mutable struct GradientProfileAnalysis{D, G, F, W, A}
     ϵ :: D
    ∇ϵ :: G
    ∇ϕ :: F
    gradient_weight :: W
    value_weight :: W
    analysis :: A
end

GradientProfileAnalysis(grid; analysis=mean, gradient_weight=1.0, value_weight=1.0) =
    GradientProfileAnalysis(CenterField(grid), FaceField(grid), FaceField(grid),
                            gradient_weight, value_weight, analysis)

GradientProfileAnalysis(; analysis=mean, gradient_weight=1.0, value_weight=1.0) =
    GradientProfileAnalysis(nothing, nothing, nothing, gradient_weight, value_weight, analysis)

function on_grid(profile::GradientProfileAnalysis, grid)
    return GradientProfileAnalysis(grid;
                                          analysis = profile.analysis,
                                   gradient_weight = profile.gradient_weight,
                                      value_weight = profile.value_weight)
end

function calculate_gradient_discrepancy!(prof, model_field, data_field)
    # Coarse grain the data
    ϵ = prof.ϵ
    set!(ϵ, data_field)

    # Calculate profients of both data and discrepancy
    ∇ϕ = prof.∇ϕ
    ∇ϵ = prof.∇ϵ
    ∂z!(∇ϵ, ϵ)
    ∂z!(∇ϕ, model_field)

    for i in eachindex(ϵ)
        @inbounds ϵ[i] = (ϵ[i] - model_field[i])^2
        @inbounds ∇ϵ[i] = (∇ϵ[i] - ∇ϕ[i])^2 # includes bottom boundary value, which will later be ignored.
    end

    # Top boundary contribution (ignored for now)
    #N = d.grid.N
    #@inbounds ∇d[N+1] = (∇d[N+1] - ∇ϕ[N+1])^2

    return nothing
end

"""
    analyze_profile_discrepancy(prof::GradientProfileAnalysis, model_field, data_field)

Calculates the discrepencies between both values and gradients of model and data fields,
and returns an analysis of the two discrepancy profiles.
"""
function analyze_profile_discrepancy(prof::GradientProfileAnalysis, model_field, data_field)
    calculate_gradient_discrepancy!(prof, model_field, data_field)

    # Calculate analysis on gradient, excluding boundary points.
    return prof.analysis(prof.ϵ) + prof.gradient_weight * prof.analysis(prof.∇ϵ.data[2:end-1])
end

#
# Loss function utils
#

@inline get_weight(::Nothing, field_index) = 1
@inline get_weight(weights, field_index) = @inbounds weights[field_index]

function analyze_weighted_profile_discrepancy(loss, model, data, target)
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

function evaluate!(loss, parameters, model_plus_Δt, data)

    # Initialize
    initialize_forward_run!(model_plus_Δt, data, parameters, loss.targets[1])
    @inbounds loss.time_series.data[1] = analyze_weighted_profile_discrepancy(loss, model_plus_Δt, data, loss.targets[1])

    # Calculate a loss function time-series
    for (i, target) in enumerate(loss.targets)
        run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data.t[target])

        @inbounds loss.time_series.data[i] =
            analyze_weighted_profile_discrepancy(loss, model_plus_Δt, data, target)
    end

    return nothing
end

function estimate_weights(profile::ValueProfileAnalysis, data::TruthData, fields, targets, relative_weights)
    max_variances = [mean_variance(data, field; targets=targets) for field in fields]
    weights = [1/σ for σ in max_variances]

    if relative_weights != nothing
        weights .*= relative_weights
    end

    return weights
end

function estimate_weights(profile::GradientProfileAnalysis, data::TruthData, fields, targets, relative_weights)
    gradient_weight = profile.gradient_weight
    value_weight = profile.value_weight

    @warn "Dividing the gradient weight of profile by height(data.grid) = $(height(data.grid))"
    gradient_weight = profile.gradient_weight = gradient_weight / height(data.grid)

    max_variances = [max_variance(data, field, targets) for field in fields]
    #max_gradient_variances = [max_gradient_variance(data, field, targets) for field in fields]

    weights = zeros(length(fields))
    for i in length(fields)
        σ = max_variances[i]
        #ς = max_gradient_variances[i]
        weights[i] = 1/σ * (value_weight + gradient_weight / height(data.grid))
    end

    if relative_weights != nothing
        weights .*= relative_weights
    end

    max_variances = [max_variance(data, field, targets) for field in fields]
    weights = [1/σ for σ in max_variances]

    if relative_weights != nothing
        weights .*= relative_weights
    end

    return weights
end

function init_negative_log_likelihood(model::ParameterizedModel, data::TruthData, first_target, last_target,
                                      fields, relative_weights)

    grid = model.model.grid

    profile_analysis = ValueProfileAnalysis(grid)
    # Create loss function and negative-log-likelihood object
    last_target = last_target === nothing ? length(data) : last_target
    targets = first_target:last_target
    profile_analysis = on_grid(profile_analysis, grid)
    weights = estimate_weights(profile_analysis, data, fields, targets, relative_weights)

    # Create loss function and LossFunction
    loss = LossFunction(model.model, data, fields=fields, targets=targets, weights=weights,
                        time_series = TimeSeriesAnalysis(data.t[targets], TimeAverage()),
                        profile = profile_analysis)

    loss = LossFunction(model, data, loss)

    return loss
end

#
# Miscellanea
#

function max_variance(data, loss::LossFunction)
    max_variances = zeros(length(loss.fields))
    for (ifield, field) in enumerate(loss.fields)
        max_variances[ifield] = get_weight(loss.weights, ifield) * max_variance(data, field, loss.targets)
    end
    return max_variances
end


function mean_variance(data, loss::LossFunction)
    mean_variance = zeros(length(loss.fields))
    for (ifield, field) in enumerate(loss.fields)
        mean_variance[ifield] = get_weight(loss.weights, ifield) * mean_variance(data, field, loss.targets)
    end
    return mean_variances
end
