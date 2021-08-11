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

function LossFunction(model, data; fields,
                          targets = 1:length(data.t),
                          weights = nothing,
                      time_series = TimeSeriesAnalysis(data.t[targets], TimeAverage()),
                          profile = ValueProfileAnalysis(model.grid)
                      )

    return LossFunction(targets, fields, weights, time_series, profile)
end

function (loss::LossFunction)(θ, model, data)
    evaluate!(loss, θ, model, data)
    return loss.time_series.analysis(loss.time_series.data, loss.time_series.time)
end

mutable struct LossContainer{M<:ParameterizedModel, D<:TruthData, L<:LossFunction} <: ALC
    model :: M
     data :: D
     loss :: L
end

(lc::LossContainer)(θ) = lc.loss(θ, lc.model, lc.data)

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

#
# Ensemble loss function
#

# set!(model, parameters, ensemble)
# Ensemble calibration
# `Batch` here refers to all of the physical scenarios to be calibrated to
const BatchLossFunction = Vector{<:LossFunction}

mutable struct EnsembleLossContainer{D, W, M, L, E} <: ALC
    data_batch :: D
       weights :: W # scenario weights
         model :: M
    loss_batch :: L
         error :: E
end

function EnsembleLossContainer(model, data_batch, loss_batch; weights=[1.0 for b in batch])
    return EnsembleLossContainer(data_batch, weights, model, loss_batch, zeros(length(data_batch)))
end

function (el::EnsembleLossContainer)(θ::Vector{<:FreeParameters})

    set!(el.model, θ)

    # iterate the model

    ensemble_size = ensemble_size(el.model)
    el.error .= 0

    # @inbounds begin
    #     Base.Threads.@threads for j = 1:length(el.data_batch)

            # mean error across all ensemble members for this data case

    evaluate!(el, θ, el.model, data_batch)
    
    loss.time_series.analysis(loss.time_series.data, loss.time_series.time) / ensemble_size

        #     el.error[i] = el.weights[i] * el.batch[i](θ)
        # end
    # end
    return sum(el.error)
end

function calculate_value_discrepancy!(value, model_field, data_field)
    discrepancy = value.discrepancy

    centered_data = CenterField(model_field.grid)
    set!(centered_data, data_field)

    centered_model = CenterField(model_field.grid)
    set!(centered_model, model_field)

    N_ens = model_field.grid.Nx

    interior(discrepancy) .= (interior(centered_data) .- interior(centered_model)) .^ 2

    return nothing
end

"""
    analyze_profile_discrepancy(value, model_field, data_field)

Calculates the discrepancy between model and data field values, and returns an
analysis of the discrepancy profile.
"""
function analyze_profile_discrepancy(value, model_field, data_field)
    calculate_value_discrepancy!(value, model_field, data_field) # MSE for each grid element
    return value.analysis(value.discrepancy) # e.g. mean or ensemble_mean of discrepancy field
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



function new_field(field_name, field_data, grid)

    field_name == :u && return XFaceField(simulation_grid, field_data)
    field_name == :v && return YFaceField(simulation_grid, field_data)
    field_name == :b && return CenterField(simulation_grid, field_data)
    field_name == :e && return CenterField(simulation_grid, field_data)

end

function analyze_weighted_profile_discrepancy(loss, model, data_batch::BatchTruthData, target)

    N_ens = ensemble_size(model)
    total_discrepancy = zeros(N_ens)

    field_names = Tuple(loss.fields)

    for (field_index, field_name) in enumerate(field_names)
        model_field = getproperty(model, field_name)
        # data_field = getproperty(data, field_name)[target]

        field_data = many_columns_interior(td_batch, field_name, target, N_ens)
        data_field = new_field(field_name, field_data, model_field.grid)

        # Calculate the per-field profile-based disrepancy
        field_discrepancy = analyze_profile_discrepancy(loss.profile, model_field, data_field)

        # Accumulate weighted profile-based disrepancies in the total discrepancyor
        total_discrepancy .+= get_weight(loss.weights, field_index) * field_discrepancy # accumulate discrepancyor
    end

    return nan2inf.(total_discrepancy)
end

function evaluate!(loss, parameters, model_plus_Δt, data::TruthData)

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

function evaluate!(loss, parameters, model_plus_Δt, data_batch::BatchTruthData)

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

function init_loss_function(model::ParameterizedModel, data::TruthData, first_target, last_target,
                                      fields, relative_weights; analysis = mean)

    grid = model.grid

    profile_analysis = ValueProfileAnalysis(grid, analysis = analysis)
    last_target = last_target === nothing ? length(data) : last_target
    targets = first_target:last_target
    profile_analysis = on_grid(profile_analysis, grid)
    weights = estimate_weights(profile_analysis, data, fields, targets, relative_weights)

    loss_function = LossFunction(model, data, fields=fields, targets=targets, weights=weights,
                        time_series = TimeSeriesAnalysis(data.t[targets], TimeAverage()),
                        profile = profile_analysis)

    return loss_function
end

function init_loss_function(model::ParameterizedModel, data_batch::BatchTruthData, 
                            first_targets, last_targets, fields, relative_weights)

    loss_batch = [init_loss_function(model, data_batch[i], 
                            first_targets[i], 
                            last_targets[i], 
                            fields[i], 
                            [relative_weights[fld] for fld in fields[i]];
                            analysis = ensemble_mean) for i in eachindex(data_batch)]

    return loss_batch
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
