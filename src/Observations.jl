module Observations

using Oceananigans
using Oceananigans: short_show, fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields
using Oceananigans.Utils: SpecifiedTimes
using JLD2

import Oceananigans.Fields: set!

abstract type AbstractObservation end

include("normalization.jl")

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

function OneDimensionalTimeSeries(path; field_names, normalize=IdentityNormalization, times=nothing)
    field_names = tupleit(field_names)
    fields = NamedTuple(name => FieldTimeSeries(path, string(name); times) for name in field_names)
    grid = first(fields).grid
    times = first(fields).times

    # validate_data(fields, grid, times) # might be a good idea to validate the data...
    file = jldopen(path)
    metadata = NamedTuple(Symbol(group) => read_group(file[group]) for group in filter(n -> n ∉ not_metadata_names, keys(file)))
    close(file)

    normalization = Dict(name => normalize(fields[name]) for name in keys(fields))

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

function set!(model, ts::OneDimensionalTimeSeries, index=1)
    # Set initial condition
    for name in keys(fields(model))
        field = fields(model)[name]
        set!(field, initial_condition(ts, name, index))
    end

    return nothing
end

struct FieldTimeSeriesCollector{G, D, F, T}
    grid :: G
    field_time_serieses :: D
    collected_fields :: F
    times :: T
end

"""
    FieldTimeSeriesCollector(fields)

Returns a `FieldTimeSeriesCollector` for `fields` of `simulation`.

`fields` is a `NamedTuple` of `AbstractField`s that are to be collected.
"""
function FieldTimeSeriesCollector(collected_fields, times; architecture=CPU())

    grid = collected_fields[1].grid
    field_time_serieses = Dict{Symbol, Any}()

    for name in keys(collected_fields)
        field = collected_fields[name]
        LX, LY, LZ = location(field)
        field_time_series = FieldTimeSeries{LX, LY, LZ}(architecture, field.grid, times)
        field_time_serieses[name] = field_time_series
    end

    # Convert to NamedTuple
    field_time_serieses = NamedTuple(name => field_time_serieses[name] for name in keys(collected_fields))

    return FieldTimeSeriesCollector(grid, field_time_serieses, collected_fields, times)
end

function (collector::FieldTimeSeriesCollector)(simulation)
    for field in collector.collected_fields
        compute!(field)
    end

    current_time = simulation.model.clock.time
    time_index = findfirst(t -> t >= current_time, collector.times)

    for name in keys(collector.collected_fields)
        field_time_series = collector.field_time_serieses[name]
        set!(field_time_series[time_index], collector.collected_fields[name])
    end

    return nothing
end

function initialize_simulation!(simulation, ts::OneDimensionalTimeSeries, time_series_collector, time_index=1)
    set!(simulation.model, ts, time_index) 

    initial_time = ts.times[time_index]
    simulation.model.clock.time = initial_time
    simulation.model.clock.iteration = 0

    # Zero out time series data
    for time_series in time_series_collector.field_time_serieses
        time_series.data .= 0
    end

    simulation.callbacks[:data_collector] = Callback(time_series_collector, SpecifiedTimes(ts.times...))

    simulation.stop_time = ts.times[end]

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