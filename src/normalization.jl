
function mean_mean(field_time_series)

    mean_time_series = mean.(interior.(field_time_series))
    
    return mean(mean_time_series)
end

function variance(field)

    field_mean = mean(field)

    variance = mean([(value - field_mean)^2 for value in field])

    return variance
end

function mean_variance(field_time_series)

    variance_time_series = variance.(interior.(field_time_series))

    return mean(variance_time_series)
end


#
# Normalization functionality for forward map output
#

abstract type AbstractNormalization end

struct Identity <: AbstractNormalization end

normalize(::Identity, field) = field

struct ZScore{T} <: AbstractNormalization
    μ :: T
    σ :: T

    function ZScore(field_time_series)

        μ = mean_mean(field_time_series)
        σ = sqrt(mean_variance(field_time_series))

        return new(μ, σ)
    end
end

normalize(::ZScore, field) = (field .- μ) ./ σ
