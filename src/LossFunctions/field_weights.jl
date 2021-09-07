function estimate_weights(profile::ValueProfileAnalysis, data::TruthData, relative_weights)
    mean_variances = [mean_variance(data, field) for field in data.relevant_fields]
    weights = [1/σ for σ in mean_variances]

    if !isnothing(relative_weights)
        weights .*= relative_weights
    end

    return weights
end

function estimate_weights(profile::GradientProfileAnalysis, data::TruthData, relative_weights)
    gradient_weight = profile.gradient_weight
    value_weight = profile.value_weight

    @warn "Dividing the gradient weight of profile by data.grid.Lz = $(data.grid.Lz))"
    gradient_weight = profile.gradient_weight = gradient_weight / data.grid.Lz

    fields = data.relevant_fields

    max_variances = [max_variance(data, field) for field in fields]
    #max_gradient_variances = [max_gradient_variance(data, field) for field in fields]

    weights = zeros(length(fields))
    for i in 1:length(fields)
        σ = max_variances[i]
        #ς = max_gradient_variances[i]
        weights[i] = 1/σ * (value_weight + gradient_weight / data.grid.Lz)
    end

    if relative_weights != nothing
        weights .*= relative_weights
    end

    max_variances = [max_variance(data, field) for field in fields]
    weights = [1/σ for σ in max_variances]

    if !isnothing(relative_weights)
        weights .*= relative_weights
    end

    return weights
end