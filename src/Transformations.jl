module Transformations

using Statistics

using Oceananigans.Fields: interior
using Oceananigans.OutputReaders: FieldTimeSeries

#####
##### Identity normalization
#####

struct Transformation{T, X, N}
    time :: T
    space :: X
    normalization :: N
end

"""
    Transformation(; time=nothing, space=nothing, normalization=nothing)

"""
Transformation(; time=nothing, space=nothing, normalization=nothing) =
    Transformation(time, space, normalization)

"""
    compute_time_transformation(user_time_transformation, fts)

Compute a time transformation for the field time series `fts`
given `user_time_transformation`.

By default, we include all time slices except the initial condition.
"""
function compute_time_transformation(::Nothing, fts)
    Nt = length(fts.times)
    return 2:Nt
end

compute_space_transformation(::Nothing, fts) = nothing
compute_space_transformation(indices::Tuple, fts) = indices
compute_normalization(::Nothing, transformation, fts) = nothing

function compute_transformation(transformation, field_time_series)
    time = compute_time_transformation(transformation.time, field_time_series)
    space = compute_space_transformation(transformation.space, field_time_series)
    normalization = compute_normalization(transformation.normalization, transformation, field_time_series)
    return Transformation(time, space, normalization)
end

#####
##### Time transformations
#####

time_transform(::Nothing, data) = data
time_transform(indices::UnitRange, data) = data[:, :, :, indices]

function time_transform(weights::AbstractVector, data)
    weights = reshape(weights, 1, 1, 1, length(weights))
    return data .* weights
end

#####
##### Space transformations
#####

space_transform(::Nothing, data) = data
space_transform(indices::Tuple, data) = data[indices[1], indices[2], indices[3], :]

function space_transform(weights::AbstractArray, data)
    Nx, Ny, Nz = size(weights)
    weights = reshape(weights, Nx, Ny, Nz, 1)
    return data .* weights
end

#####
##### Normalizations
#####

abstract type AbstractNormalization end

# Convenience is queen
function compute_transformation(normalization::Union{Nothing, AbstractNormalization}, field_time_series)
    transformation = Transformation(; normalization)
    return compute_transformation(transformation, field_time_series)
end

normalize!(data, ::Nothing) = data

#####
##### ZScore, kinda like Normal...
#####

struct ZScore{T} <: AbstractNormalization
    μ :: T
    σ :: T
end

ZScore() = ZScore(nothing, nothing) # stub for user-interface

"""
    ZScore(field_time_series::FieldTimeSeries)

Return the `ZScore` normalization of a `FieldTimeSeries` after computing
its mean and its variance.
"""
function ZScore(data)
    @assert size(data, 1) == 1 # we're going to assume this
    @assert ndims(data) == 2 # we're also going to assume this

    data = dropdims(data, dims=1)
    μ = mean(data)
    σ = sqrt(cov(data; corrected=false))

    if σ == 0
        @warn "Your data has zero variance --- just sayin'! I'm setting ZScore σ = 1."
        σ = one(μ)
    end

    return ZScore(μ, σ)
end

compute_normalization(zscore::ZScore, fts) = zscore

"""Compute ZScore on time- and space-transformed field time series data."""
function compute_normalization(::ZScore{Nothing}, transformation, field_time_series)
    time_space_transformation = Transformation(transformation.time,
                                               transformation.space,
                                               nothing)

    data = transform_field_time_series(time_space_transformation, field_time_series)

    return ZScore(data)
end

function normalize!(data, normalization::ZScore)
    μ, σ = normalization.μ, normalization.σ
    @. data = (data - μ) / σ
    return nothing
end

#####
##### Like ZScore, but maybe less important
#####

struct RescaledZScore{T, Z} <: AbstractNormalization
    scale :: T
    zscore :: Z
end

RescaledZScore(scale) = RescaledZScore(scale, nothing)

compute_normalization(r::RescaledZScore, transformation, fts) =
    RescaledZScore(r.scale, compute_normalization(r.zscore, transformation, fts))

function normalize!(data, normalization::RescaledZScore)
    normalize!(data, normalization.zscore)
    @. data *= normalization.scale
    return nothing
end

#####
##### Transform it!
#####

function transform_field_time_series(transformation::Transformation,
                                     field_time_series::FieldTimeSeries)

    copied_data = Array(interior(field_time_series))
    time_transformed_data = time_transform(transformation.time, copied_data)
    time_space_transformed_data = space_transform(transformation.space, time_transformed_data)
    normalize!(time_space_transformed_data, transformation.normalization)

    # Reshape data to 2D array with size (Nx, :)
    Nens = size(time_space_transformed_data, 1)
    Ndata = size(time_space_transformed_data)[2:4]
    reshaped_data = reshape(time_space_transformed_data, Nens, prod(Ndata))

    return reshaped_data
end

end # module

