using Oceananigans.OutputReaders: FieldTimeSeries
using Statistics

import Oceananigans.Fields: interior
import Oceananigans.Grids: topology, interior_parent_indices

function variance(field)
    field_mean = mean(field)
    variance = mean(f_ijk -> (f_ijk - field_mean)^2, field)
    return variance
end

#=
function Base.iterate(field_time_series::FieldTimeSeries, state)
    if state >= length(field_time_series.times)
        return nothing
    else
        return field_time_series[state+1], state+1
    end
end
=#

mean_variance(field_time_series) = mean((variance(field_time_series[i]) for i = 1:length(field_time_series.times)))

#####
##### Normalization functionality for forward map output
#####

abstract type AbstractNormalization end

struct IdentityNormalization <: AbstractNormalization end

IdentityNormalization(field_time_series) = IdentityNormalization()

normalize!(field, ::IdentityNormalization) = nothing

struct ZScore{T} <: AbstractNormalization
    μ :: T
    σ :: T
end

function normalize!(field, z_score::ZScore)
    field .-= z_score.μ
    if z_score.σ != 0
        field ./= z_score.σ
    end
    return nothing
end

#=
"Returns a view of `f` that excludes halo points."
@inline interior(f::FieldTimeSeries{X, Y, Z}) where {X, Y, Z} =
    view(parent(f.data),
         interior_parent_indices(X, topology(f, 1), f.grid.Nx, f.grid.Hx),
         interior_parent_indices(Y, topology(f, 2), f.grid.Ny, f.grid.Hy),
         interior_parent_indices(Z, topology(f, 3), f.grid.Nz, f.grid.Hz),
         :)
=#

function ZScore(field_time_series)
    μ = mean(interior(field_time_series))
    σ = sqrt(mean_variance(field_time_series))
    return ZScore(μ, σ)
end