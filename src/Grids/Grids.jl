module Grids

using ..OceanTurbulenceParameterEstimation

using Oceananigans
using Oceananigans.Grids: Flat, Bounded, Periodic, 
                          Face, Center,
                          AbstractGrid,
                          RegularRectilinearGrid

using JLD2

export OneDimensionalEnsembleGrid

include("utils.jl")
include("one_dimensional_ensemble_grid.jl")

end #module
