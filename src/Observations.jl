module Observations

using Oceananigans
using Oceananigans: short_show, fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields
using JLD2

import Oceananigans.Fields: set!

abstract type AbstractObservation end

"""
    OneDimensionalTimeSeries{F, G, T, P, M} <: AbstractObservation

A time series of horizontally-averaged observational or LES data
gridded as Oceananigans fields.
"""
struct OneDimensionalTimeSeries{F, G, T, P, M, N} <: AbstractObservation
          fields :: F
            grid :: G
           times :: T
            path :: P
        metadata :: M
   normalization :: N
end

obs_str(ts::OneDimensionalTimeSeries) = "OneDimensionalTimeSeries of $(keys(ts.fields)) on $(short_show(ts.grid))"

tupleit(t) = try
    Tuple(t)
catch
    tuple(t)
end

const not_metadata_names = ("serialized", "timeseries")

read_group(group::JLD2.Group) = NamedTuple(Symbol(subgroup) => read_group(group[subgroup]) for subgroup in keys(group))
read_group(group) = group

function OneDimensionalTimeSeries(path; field_names; normalize=Identity())
    field_names = tupleit(field_names)
    fields = NamedTuple(name => FieldTimeSeries(path, string(name)) for name in field_names)
    grid = fields[1].grid
    times = fields[1].times

    # validate_data(fields, grid, times) # might be a good idea to validate the data...
    file = jldopen(path)
    metadata = NamedTuple(Symbol(group) => read_group(file[group]) for group in filter(n -> n ∉ not_metadata_names, keys(file)))
    close(file)

    normalization = [normalize.body(field_time_series) for field_time_series in values(fields)]

    return OneDimensionalTimeSeries(fields, grid, times, path, metadata, normalization)
end

#####
##### set! for simulation models and observations
#####

default_initial_condition(ts) = 0

function initial_condition(ts, field_name, time_index)
    if field_name in keys(ts.fields)
        return ts.fields[field_name][time_index]
    else
        return default_initial_condition(ts)
    end
end

function set!(model::AbstractField, ts::OneDimensionalTimeSeries, index=1)
    # Set initial condition
    for name in keys(fields(model))
        field = fields(model)[name]
        set!(field, initial_condition(ts, name, index))
    end

    return nothing
end

function initialize_simulation!(simulation, ts::OneDimensionalTimeSeries, time_index=1)
    set!(simulation.model, ts, time_index) 

    initial_time = ts.times[time_index]
    simulation.model.clock.time = initial_time
    simulation.model.clock.iteration = 0

    return nothing
end

Base.show(io::IO, ts::OneDimensionalTimeSeries) =
    print(io, "OneDimensionalTimeSeries with fields $(propertynames(ts.fields))", '\n',
              "├── times: $(ts.times)", '\n',    
              "├── grid: $(short_show(ts.grid))", '\n',
              "├── path: \"$(ts.path)\"", '\n',
              "├── metadata: $(keys(ts.metadata))", '\n',
              "└── normalization: $(short_show(ts.normalization))")

end # module