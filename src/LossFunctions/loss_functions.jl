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

mutable struct EnsembleLossContainer{D, F, FT, LT, M, T, L} <: ALC
    data_batch :: D
 field_weights :: F # scenario weights
 first_targets :: FT
  last_targets :: LT
         model :: M
   time_series :: T
       profile :: L
end

function EnsembleLossContainer(model, data_batch, fields, LESdata; data_weights=[1.0 for b in data_batch], relative_weights)

    last_target(data, last) = isnothing(last) ? length(data) : last_target 

    first_targets = getproperty.(values(LESdata), :first)
    last_targets = getproperty.(values(LESdata), :last)
    last_targets = last_target.(data_batch, last_targets)
    all_targets = [first_targets[i]:last_targets[i] for i in 1:length(first_targets)]
    
    profile = ValueProfileAnalysis(model.grid, analysis = column_mean)

    field_weights = Dict(f => [] for f in [:u, :v, :b, :e])
    for (i, data) in enumerate(data_batch)
        data_fields = fields[i] # e.g. (:b, :e)
        targets = first_targets[i]:last_targets[i]
        rw = [relative_weights[f] for f in data_fields]
        weights = estimate_weights(profile, data, data_fields, targets, rw) # e.g. (1.0, 0.5)

        for (j, field_name) in enumerate(data_fields)
            push!(field_weights[field_name], weights[j] * data_weights[i])
        end

        for field_name in keys(field_weights)
            field_name ∉ data_fields &&
                push!(field_weights[field_name], 0)
        end 
    end

    time_series = [EnsembleTimeSeriesAnalysis(data_batch[i].t[all_targets[i]], model.grid.Nx) for i in 1:length(data_batch)]

    return EnsembleLossContainer(data_batch, field_weights, first_targets, last_targets, model, time_series, profile)
end

function (el::EnsembleLossContainer)(θ::Vector{<:FreeParameters})

    set!(el.model, θ)

    # iterate the model and record discrepancy summary in `time_series`
    evaluate!(el, θ, el.model, el.data_batch)

    error = zeros(model.grid.Nx)
    
    for ts in el.time_series
        error .+= ts.analysis(ts.data) / ensemble_size
    end

    return error
end

function calculate_value_discrepancy!(value, model_field, data_field)
    discrepancy = value.discrepancy

    centered_data = CenterField(model_field.grid)
    set!(centered_data, data_field)

    centered_model = CenterField(model_field.grid)
    set!(centered_model, model_field)

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
    return value.analysis(value.discrepancy) # e.g.column_mean of discrepancy field
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

function analyze_weighted_profile_discrepancy(loss::EnsembleLossContainer, model, data_batch::BatchTruthData)

    total_discrepancy = zeros(model.grid.Nx, model.grid.Ny, 1)

    for field_name in [:u, :v, :b, :e]
        model_field = getproperty(model, field_name)

        # compensate for setting model time index 1 to to index `first_target` in data.
        data_indices = target .+ el.first_targets .- 1

        field_data = column_ensemble_interior(td_batch, field_name, data_indices, N_ens)
        data_field = new_field(field_name, field_data, model_field.grid)

        # Calculate the per-field profile-based discrepancy
        field_discrepancy = analyze_profile_discrepancy(loss.profile, model_field, data_field)

        # Accumulate weighted profile-based discrepancies in the total discrepancyor
        total_discrepancy += el.field_weights[field_name]' .* field_discrepancy # accumulate discrepancyor
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

allsame(x) = all(y -> y ≈ first(x), x)
Δt(data) = data.t[2:end] .- data.t[1:end-1]

function evaluate!(el, parameters, model_plus_Δt, data_batch::BatchTruthData)

    @assert all([allsame(Δt(data)) for data in data_batch]) "Simulation time steps are not uniformly spaced."
    @assert allsame([Δt(data)[1] for data in data_batch]) "Time step differs between simulations."

    # Initialize
    initialize_forward_run!(model_plus_Δt, data_batch, parameters, el.first_targets)
    
    simulation_lengths = length.(el.targets)

    # Calculate a loss function time-series
    for (i, target) in 1:maximum(simulation_lengths)
    
        run_until!(model_plus_Δt.model, model_plus_Δt.Δt, data_batch[1].t[target])

        discrepancy = analyze_weighted_profile_discrepancy(el, model_plus_Δt, data_batch, target)

        for (j, ts) in enumerate(el.time_series)
            if target <= length(ts.time)
                # `ts.data` is N_ensemble x N_timesteps; `discrepancy` is N_ensemble x N_cases x 1
                ts.data[:, i] .= discrepancy[:, j, 1]
            end
        end

    end

    return nothing
end

function estimate_weights(profile::ValueProfileAnalysis, data::TruthData, fields, targets, relative_weights)
    mean_variances = [mean_variance(data, field; targets=targets) for field in fields]
    weights = [1/σ for σ in mean_variances]

    if !isnothing(relative_weights)
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
    for i in 1:length(fields)
        σ = max_variances[i]
        #ς = max_gradient_variances[i]
        weights[i] = 1/σ * (value_weight + gradient_weight / height(data.grid))
    end

    if relative_weights != nothing
        weights .*= relative_weights
    end

    max_variances = [max_variance(data, field, targets) for field in fields]
    weights = [1/σ for σ in max_variances]

    if !isnothing(relative_weights)
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
