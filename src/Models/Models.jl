module Models

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.Grids
using ..OceanTurbulenceParameterEstimation.Data

using Oceananigans
using Oceananigans: AbstractModel
using Oceananigans.Fields
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure, AbstractEddyViscosityClosure
using Oceananigans.Models

using Printf

import Oceananigans.Fields: interpolate
import Base: size
import StaticArrays: FieldVector
import OceanTurbulenceParameterEstimation.Data: set!

export
       get_model_field,
       set!,

       # ensemble_model.jl
       EnsembleModel, 
       ensemble_size, batch_size,
       initialize_forward_run!,

       # free_parameters.jl
       DefaultFreeParameters,
       get_free_parameters,
       FreeParameters,
       @free_parameters

#
# AbstractModel extensions
#

function get_model_field(m::AbstractModel, p)
    p ∈ propertynames(m.tracers) && return m.tracers[p]
    p ∈ propertynames(m.velocities) && return m.velocities[p]
    @error "$p is not a valid field name"
end

"""
set!(model; kwargs...)

Set velocity and tracer fields of `model` given data defined on `grid`.
The keyword arguments `kwargs...` take the form `name=data`, where `name` refers
to one of the fields of `model.velocities` or `model.tracers`, and the `data`
may be an array, a function with arguments `(x, y, z)`, or any data type for which a
`set!(ϕ::AbstractField, data)` function exists.

Equivalent to `Oceananigans.Models.HydrostaticFreeSurfaceModels.set!` but where the
data in `kwargs...` exist on an arbitrary-resolution grid `grid`
and may need to be interpolated to the model grid `model.grid`.
"""
function set!(model; kwargs...)

    for (fldname, value) in kwargs

        # Get the field `ϕ` corresponding to `fldname`.
        if fldname ∈ propertynames(model.velocities)
            ϕ = getproperty(model.velocities, fldname)
        elseif fldname ∈ propertynames(model.tracers)
            ϕ = getproperty(model.tracers, fldname)
        elseif fldname ∈ propertynames(model.free_surface)
            ϕ = getproperty(model.free_surface, fldname)
        else
            throw(ArgumentError("name $fldname not found in model.velocities or model.tracers."))
        end

        set!(ϕ, value)
    end

    return nothing
end

include("ensemble_model.jl")
include("free_parameters.jl")

include("CATKEVerticalDiffusivityModel/CATKEVerticalDiffusivityModel.jl")

using .CATKEVerticalDiffusivityModel

end #module