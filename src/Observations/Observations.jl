module Observations

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.Grids: ColumnEnsembleGrid

using Oceananigans
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields
using JLD2

abstract type AbstractTimeSeries end

const AbstractTimeSeriesBatch = Vector{<:AbstractTimeSeries}

export # truth_data.jl
       OneDimensionalTimeSeries, 
       OneDimensionalTimeSeriesBatch,
       column_ensemble_interior,

       # set_field.jl
       set!

include("file_wrangling.jl")
include("truth_data.jl")
include("set_field.jl")

end #module