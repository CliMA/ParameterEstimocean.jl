module ModelsAndData

using ..OceanTurbulenceParameterEstimation

import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Fields: interpolate, AbstractField
import Oceananigans.Models: AbstractModel, HydrostaticFreeSurfaceModel
import Oceananigans.Grids: AbstractGrid
import Base: length, size, getproperty
import StaticArrays: FieldVector

using Oceananigans
using Oceananigans: AbstractEddyViscosityClosure
using Oceananigans.Fields: CenterField, AbstractDataField
using Oceananigans.Grids: Flat, Bounded, Periodic, RegularRectilinearGrid, Face, Center
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure

using OrderedCollections, Printf, JLD2
using Base: nothing_sentinel

export
       initialize_forward_run!,
       EnsembleModel, EnsembleGrid,
       ensemble_size, batch_size,
    #    getproperty,
       get_model_field,

       # LESbrary_paths.jl
       LESbrary,
       TwoDaySuite, FourDaySuite, SixDaySuite, GeneralStrat,

       # grids.jl
       ColumnEnsembleGrid, XYZGrid,

       # data.jl
       TruthData, BatchTruthData,

       # set_fields.jl
       set!,
       column_ensemble_interior,

       # free_parameters.jl
       DefaultFreeParameters,
       get_free_parameters,
       FreeParameters,
       @free_parameters

#
# AbstractModel extension
#

function get_model_field(m::AbstractModel, p)
    p ∈ propertynames(m.tracers) && return m.tracers[p]
    p ∈ propertynames(m.velocities) && return m.velocities[p]
    @error "$p is not a valid field name"
end

# function Base.getproperty(m::AbstractModel, ::Val{p}) where p

#     p ∈ propertynames(m.tracers) && return m.tracers[p]

#     p ∈ propertynames(m.velocities) && return m.velocities[p]

#     return getproperty(m, p)

# end

const EnsembleGrid = RegularRectilinearGrid{<:Any, Flat, Flat, Bounded}
const EnsembleModel = HydrostaticFreeSurfaceModel{TS, E, A, S, <:EnsembleGrid, T, V, B, R, F, P, U, C, Φ, K, AF} where {TS, E, A, S, T, V, B, R, F, P, U, C, Φ, K, AF}

ensemble_size(model::EnsembleModel) = model.grid.Nx
batch_size(model::EnsembleModel) = model.grid.Ny

include("file_wrangling.jl")
include("lesbrary_paths.jl")
include("grids.jl")
include("data.jl")
include("set_fields.jl")
include("free_parameters.jl")

"""
    initialize_forward_run!(model, data_batch::BatchTruthData, params, time_indices::Vector)

Set columns of each field in `model` to the corresponding profile columns in `data_batch`, 
where every field column in `model` that corresponds to the ith `TruthData` object in `data_batch`
is set to the field column in `data_batch[i]` at `time_indices[i]`.
"""

function initialize_forward_run!(model::EnsembleModel, data_batch::BatchTruthData, params, time_indices::Vector)
    set!(model, params)
    set!(model, data_batch, time_indices)
    model.clock.time = 0.0
    model.clock.iteration = 0
    return nothing
end

end #module