using Oceananigans.OutputReaders: FieldTimeSeries
using Statistics

import Oceananigans.Fields: interior
import Oceananigans.Grids: topology, interior_parent_indices

function variance(field)
    field_mean = mean(interior(field))
    variance = mean(f_ijk -> (f_ijk - field_mean)^2, interior(field))
    return variance
end

mean_variance(field_time_series) = mean((variance(field_time_series[i])
                                         for i = 1:length(field_time_series.times)))

#####
##### Identity normalization
#####

struct IdentityNormalization end

"""
    compute_normalization_properties!(normalization, field_time_series)

Compute `normalization` properties for `field_time_series`.
"""
compute_normalization_properties!(::IdentityNormalization, fts) =
    IdentityNormalization()

"""
    normalize!(field, normalization)

Normalize `field` using `normalization`.
"""
normalize!(field, ::IdentityNormalization) = nothing

#####
##### ZScore normalization
#####

struct ZScore{T}
    μ :: T
    σ :: T
end

ZScore() = ZScore(nothing, nothing) # stub for user-interface

"""
    ZScore(field_time_series::FieldTimeSeries)

Return the `ZScore` normalization of a `FieldTimeSeries` after computing
its mean and its variance.
"""
function ZScore(field_time_series::FieldTimeSeries)
    μ = mean(interior(field_time_series))
    σ = sqrt(mean_variance(field_time_series))

    if σ == 0
        @warn("Field data seems to be all zeros --- just sayin'. Setting " *
              "ZScore standard deviation to 1.")

        σ = one(μ)
    end

    return ZScore(μ, σ)
end

compute_normalization_properties!(::ZScore, fts) = ZScore(fts)

function normalize!(field, normalization::ZScore)
    μ, σ = normalization.μ, normalization.σ
    @. field = (field - μ) / σ
    return nothing
end

#####
##### ZScore normalization
#####

struct RescaledZScore{T, Z}
    scale :: T
    zscore :: Z
end

RescaledZScore(scale) = RescaledZScore(scale, nothing)

compute_normalization_properties!(r::RescaledZScore, fts) =
    RescaledZScore(r.scale, ZScore(fts))

function normalize!(field, normalization::RescaledZScore)
    normalize!(field, normalization.zscore)
    @. field *= normalization.scale
    return nothing
end

