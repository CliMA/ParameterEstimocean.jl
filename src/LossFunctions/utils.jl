
function profile_mean(data, field_name)
    total_mean = 0.0
    fields = getproperty(data, field_name)
    data = Oceananigans.Fields.interior(field)

    for target in data.targets
        field = fields[target]
        field_mean = mean(data)
        total_mean += field_mean
    end

    return total_mean / length(data.targets)
end

function variance(field::AbstractDataField)

    data = Oceananigans.Fields.interior(field)
    field_mean = mean(data)

    variance = zero(eltype(field))
    for j in eachindex(data)
        variance += (data[j] - field_mean)^2
    end

    # Average over the number of elements in the array
    return variance / length(data)
end

# Returns the field variance for each target in `data.targets`
function variances(data::TruthData, field_name)

    variances = zeros(length(data.targets))
    fields = getproperty(data, field_name)

    for (i, target) in enumerate(data.targets)
        variances[i] = variance(fields[target])
    end

    return variances
end

mean_variance(data::TruthData, field_name)   = mean(variances(data::TruthData, field_name))
max_variance(data::TruthData, field_name) = maximum(variances(data::TruthData, field_name))
mean_std(data::TruthData, field_name)  = mean(sqrt.(variances(data::TruthData, field_name)))

nan2inf(err) = isnan(err) ? Inf : err

function trapz(f, t)
    @inbounds begin
        integral = zero(eltype(t))
        for i = 2:length(t)
            integral += (f[i] + f[i-1]) * (t[i] - t[i-1])
        end
    end
    return integral
end