module Data

using ..OceanTurbulenceParameterEstimation
using ..OceanTurbulenceParameterEstimation.Grids: ColumnEnsembleGrid

using Oceananigans
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields
using JLD2

export # truth_data.jl
       OneDimensionalTimeSeries, 
       OneDimensionalTimeSeriesBatch,
       column_ensemble_interior,

       # set_field.jl
       set!

include("file_wrangling.jl")
include("truth_data.jl")
include("truth_observations.jl")
include("set_field.jl")

end #module