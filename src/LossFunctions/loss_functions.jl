
mutable struct LossFunction{Δ, FT, ML, F, T, L, P}
                     Δt :: Δ
          first_targets :: FT
  max_simulation_length :: ML
          field_weights :: F # scenario weights
            time_series :: T
                profile :: L
   ParametersToOptimize :: P
end

allsame(x) = all(y -> y ≈ first(x), x)
t_interval(data) = data.t[2:end] .- data.t[1:end-1]

function (loss::LossFunction)(model, data_batch, θ::Vector{<:FreeParameters})

    # iterate the model and record discrepancy summary in `time_series`
    evaluate!(loss, θ, model, data_batch)

    N_ens = ensemble_size(model)
    error = zeros((N_ens,1))
    
    for ts in loss.time_series
        data_error = ts.analysis(ts.data) / N_ens
        error .+= data_error
    end

    return error
end

(loss::LossFunction)(m, db::BatchTruthData, θ::FreeParameters) = loss(m, db, [θ for i = 1:ensemble_size(m)])
(loss::LossFunction)(m, db::BatchTruthData, θ::Vector{<:Number}) = loss(m, db, loss.ParametersToOptimize(θ))
(loss::LossFunction)(m, db::BatchTruthData, θ::Vector{<:Vector}) = loss(m, db, loss.ParametersToOptimize.(θ))
(loss::LossFunction)(m, db::BatchTruthData, θ::Matrix) = loss(m, db, [θ[:,i] for i in 1:size(θ, 2)])

function LossFunction(model, data_batch::BatchTruthData, Δt, ParametersToOptimize; data_weights=[1.0 for b in data_batch], relative_weights)

    @assert all([allsame(t_interval(data)) for data in data_batch]) "Simulation time steps are not uniformly spaced."
    @assert allsame([t_interval(data)[1] for data in data_batch]) "Time step differs between simulations."

    all_targets = getproperty.(data_batch, :targets)
    first_targets = getindex.(all_targets, 1)
    max_simulation_length = maximum(length.(all_targets))
    
    profile = ValueProfileAnalysis(model.grid, analysis = column_mean)

    field_weights = Dict(f => [] for f in [:u, :v, :b, :e])
    for (i, data) in enumerate(data_batch)
        data_fields = data.relevant_fields # e.g. (:b, :e)
        targets = all_targets[i]
        rw = [relative_weights[f] for f in data_fields]
        weights = estimate_weights(profile, data, rw) # e.g. (1.0, 0.5)

        for (j, field_name) in enumerate(data_fields)
            push!(field_weights[field_name], weights[j] * data_weights[i])
        end

        for field_name in keys(field_weights)
            field_name ∉ data_fields &&
                push!(field_weights[field_name], 0)
        end 
    end

    time_series = [EnsembleTimeSeriesAnalysis(data_batch[i].t[all_targets[i]], model.grid.Nx) for i in 1:length(data_batch)]

    return LossFunction(Δt, first_targets, max_simulation_length, field_weights, time_series, profile, ParametersToOptimize)
end

function calculate_value_discrepancy!(value, model_field, data_field)
    discrepancy = value.discrepancy

    centered_data = CenterField(model_field.grid)
    set!(centered_data, interior(data_field))

    centered_model = CenterField(model_field.grid)
    set!(centered_model, interior(model_field))

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

function new_field(field_name, field_data, grid)

    field_name == :u && return XFaceField(grid, field_data)
    field_name == :v && return YFaceField(grid, field_data)
    field_name == :b && return CenterField(grid, field_data)
    field_name == :e && return CenterField(grid, field_data)

end

function analyze_weighted_profile_discrepancy(loss::LossFunction, model, data_batch::BatchTruthData, target)

    total_discrepancy = zeros(model.grid.Nx, model.grid.Ny, 1)

    for field_name in [:u, :v, :b, :e]

        model_field = get_model_field(model, field_name)

        # compensate for setting model time index 1 to to index `first_target` in data.
        data_indices = target .+ loss.first_targets .- 1

        field_data = column_ensemble_interior(data_batch, field_name, data_indices, model.grid.Nx)
        data_field = new_field(field_name, field_data, model_field.grid)

        # Calculate the per-field profile-based discrepancy
        field_discrepancy = analyze_profile_discrepancy(loss.profile, model_field, data_field)

        # Accumulate weighted profile-based discrepancies in the total discrepancyor
        total_discrepancy .+= loss.field_weights[field_name]' .* field_discrepancy # accumulate discrepancyor
    end

    return nan2inf.(total_discrepancy)
end

function evaluate!(loss::LossFunction, parameters, model, data_batch::BatchTruthData)

    # Initialize
    initialize_forward_run!(model, data_batch, parameters, loss.first_targets)

    simulation = Simulation(model; Δt=loss.Δt, stop_time = 0.0)
    
    # Calculate a loss function time-series
    for target in 1:loss.max_simulation_length

        simulation.stop_time = data_batch[1].t[target]
    
        run!(simulation)

        discrepancy = analyze_weighted_profile_discrepancy(loss, model, data_batch, target)

        for (j, ts) in enumerate(loss.time_series)
            if target <= length(ts.time)
                # `ts.data` is N_ensemble x N_timesteps; `discrepancy` is N_ensemble x N_cases x 1
                ts.data[:, target] .= discrepancy[:, j, 1]
            end
        end

    end

    return nothing
end

#
# Miscellanea
#

function max_variance(data)
    fields = data.relevant_fields
    max_variances = zeros(length(fields))
    for (i, field) in enumerate(fields)
        max_variances[i] = get_weight(loss.weights, i) * max_variance(data, field)
    end
    return max_variances
end


function mean_variance(data)
    fields = data.relevant_fields
    mean_variance = zeros(length(fields))
    for (i, field) in enumerate(fields)
        mean_variance[i] = get_weight(loss.weights, i) * mean_variance(data, field)
    end
    return mean_variances
end
