module Grids

using ..OceanTurbulenceParameterEstimation

using Oceananigans
using Oceananigans.Grids: Flat, Bounded, Periodic, 
                          Face, Center,
                          AbstractGrid,
                          RegularRectilinearGrid

using JLD2

export ColumnEnsembleGrid, EnsembleGrid

include("utils.jl")
include("column_ensemble_grid.jl")

end #module
