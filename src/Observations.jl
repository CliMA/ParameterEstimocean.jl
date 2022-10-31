module Observations

export SyntheticObservations, BatchedSyntheticObservations, observation_times

using ..Utils: prettyvector, tupleit

using Oceananigans
using Oceananigans.Fields
using Oceananigans.Architectures

using Oceananigans: fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z
using Oceananigans.Grids: pop_flat_elements, topology, halo_size, on_architecture
using Oceananigans.TimeSteppers: update_state!, reset!
using Oceananigans.Fields: indices
using Oceananigans.Utils: SpecifiedTimes, prettytime
using Oceananigans.Architectures: arch_array, architecture
using Oceananigans.OutputWriters: WindowedTimeAverage, AveragedSpecifiedTimes

using JLD2

import Oceananigans.Fields: set!

using ParameterEstimocean.Utils: field_name_pairs
using ParameterEstimocean.Transformations: Transformation, compute_transformation

#####
##### SyntheticObservations
#####

struct SyntheticObservations{F, R, G, T, P, M, Þ}
    field_time_serieses :: F
    forward_map_names :: R
    grid :: G
    times :: T
    path :: P
    metadata :: M
    transformation :: Þ
end

"""
    SyntheticObservations(path = nothing;
                          field_names,
                          forward_map_names = field_names,
                          transformation = Transformation(),
                          times = nothing,
                          field_time_serieses = nothing,
                          architecture = CPU(),
                          regrid = nothing)

Return a time series of synthetic observations generated by Oceananigans.jl's simulations
gridded as Oceananigans.jl fields.
"""
function SyntheticObservations(path = nothing;
                               field_names,
                               forward_map_names = field_names,
                               transformation = Transformation(),
                               times = nothing,
                               field_time_serieses = nothing,
                               architecture = CPU(),
                               regrid = nothing)

    field_names = tupleit(field_names)
    forward_map_names = tupleit(forward_map_names)

    all(n ∈ field_names for n in forward_map_names) ||
        throw(ArgumentError("All of the forward map names $forward_map_names must be in field names $field_names"))

    if isnothing(field_time_serieses)
        raw_time_serieses = NamedTuple(name => FieldTimeSeries(path, string(name); times, architecture) for name in field_names)
    else
        raw_time_serieses = field_time_serieses
    end

    raw_grid = first(raw_time_serieses).grid
    times = first(raw_time_serieses).times
    boundary_conditions = first(raw_time_serieses).boundary_conditions

    if isnothing(regrid)
        field_time_serieses = raw_time_serieses
        grid = raw_grid

    else # Well, we're gonna regrid stuff
        if regrid isa Tuple
            grid = with_size(regrid, raw_grid)
        elseif regrid isa AbstractGrid
            grid = regrid
        else
            error("$regrid must be a Tuple specifying a new grid size, or a new grid!")
        end

        @info string("Regridding synthetic observations...", '\n',
                     "    original grid: ", summary(raw_grid), '\n',
                     "         new grid: ", summary(grid))

        field_time_serieses = Dict()

        # Re-grid the data in `field_time_serieses`
        for (field_name, ts) in zip(keys(raw_time_serieses), raw_time_serieses)

            #LX, LY, LZ = location(ts[1])
            LX, LY, LZ = infer_location(field_name)

            new_ts = FieldTimeSeries{LX, LY, LZ}(grid, times; boundary_conditions)
        
            # Loop over time steps to re-grid each constituent field in `field_time_series`
            for n = 1:length(times)
                regrid!(new_ts[n], ts[n])
            end
        
            field_time_serieses[field_name] = new_ts
        end

        field_time_serieses = NamedTuple(field_time_serieses)
    end

    # validate_data(fields, grid, times) # might be a good idea to validate the data...
    if !isnothing(path)
        file = jldopen(path)
        metadata = NamedTuple(Symbol(group) => read_group(file[group])
                              for group in filter(n -> n ∉ not_metadata_names, keys(file)))
        close(file)
    else
        metadata = nothing
    end

    transformation = field_name_pairs(transformation, forward_map_names, "transformation")
    transformation = Dict(name => compute_transformation(transformation[name], field_time_serieses[name])
                          for name in forward_map_names)

    return SyntheticObservations(field_time_serieses, forward_map_names,
                                 grid, times, path, metadata, transformation)
end

observation_names(observations::SyntheticObservations) = keys(observations.field_time_serieses)
forward_map_names(observations::SyntheticObservations) = observations.forward_map_names

Base.summary(observations::SyntheticObservations) =
    "SyntheticObservations of $(keys(observations.field_time_serieses)) on $(summary(observations.grid))"

function observation_times(data_path::String)
    file = jldopen(data_path)
    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]
    close(file)
    return times
end

observation_times(observation::SyntheticObservations) = observation.times

#####
##### Utility for batching observations
#####

struct BatchedSyntheticObservations{O, W}
    observations :: O
    weights :: W
end

"""
    BatchedSyntheticObservations(observations; weights)

Return a collection of `observations` with `weights`, where
`observations` is a `Vector` or `Tuple` of `SyntheticObservations`.
`weights` are unity by default.
"""
function BatchedSyntheticObservations(batched_obs; weights=Tuple(1 for o in batched_obs))
    length(batched_obs) == length(weights) ||
        throw(ArgumentError("Must have the same number of weights and observations!"))

    tupled_batched_obs = tupleit(batched_obs)
    tupled_weights = tupleit(weights)

    return BatchedSyntheticObservations(tupled_batched_obs, tupled_weights)
end

# Convenience
const SO = SyntheticObservations

BatchedSyntheticObservations(first_obs::SO, second_obs::SO, other_obs...; kw...) =
    BatchedSyntheticObservations(tuple(first_obs, second_obs, other_obs...); kw...)

batch(b::BatchedSyntheticObservations) = b
batch(obs::SyntheticObservations) = BatchedSyntheticObservations([obs])
batch(obs) = BatchedSyntheticObservations(obs)

Base.first(batch::BatchedSyntheticObservations) = first(batch.observations)
Base.lastindex(batch::BatchedSyntheticObservations) = lastindex(batch.observations)
Base.getindex(batch::BatchedSyntheticObservations, i) = getindex(batch.observations, i)
Base.length(batch::BatchedSyntheticObservations) = length(batch.observations)

Base.summary(observations::BatchedSyntheticObservations) =
    "BatchedSyntheticObservations of $(keys(first(observations).field_time_serieses)) on $(summary(first(observations).grid))"

function combine_names(observations, name_getter)
    names = Set()
    for obs in observations
        push!(names, name_getter(obs)...)
    end

    return names
end

"""
    forward_map_names(observations::BatchedSyntheticObservations)

Return a Set representing the union of all names in `observations`.
"""
forward_map_names(batch::BatchedSyntheticObservations) = combine_names(batch.observations, forward_map_names)

"""
    observation_names(observations::BatchedSyntheticObservations)

Return a Set representing the union of all names in `observations`.
"""
observation_names(batch::BatchedSyntheticObservations) = combine_names(batch.observations, observation_names)

function observation_times(batch::BatchedSyntheticObservations)
    @assert all([o.times ≈ first(batch).times for o in batch.observations]) "Observations must have the same times."
    return observation_times(first(batch))
end

#####
##### Utilities for building SyntheticObservations
#####

const not_metadata_names = ("serialized", "timeseries")

read_group(group::JLD2.Group) = NamedTuple(Symbol(subgroup) => read_group(group[subgroup]) for subgroup in keys(group))
read_group(group) = group

using Oceananigans.Grids: ZRegRectilinearGrid

function with_size(new_size, old_grid)

    old_grid isa ZRegRectilinearGrid ||
        error("Cannot remake stretched grid \n $old_grid \n with a new size!")

    topo = topology(old_grid)

    x = cpu_face_constructor_x(old_grid)
    y = cpu_face_constructor_y(old_grid)
    z = cpu_face_constructor_z(old_grid)

    # Remove elements of size and new_halo in Flat directions as expected by grid
    # constructor
    new_size = pop_flat_elements(new_size, topo)
    halo = pop_flat_elements(halo_size(old_grid), topo)

    new_grid = RectilinearGrid(architecture(old_grid), eltype(old_grid);
        size = new_size,
        x = x, y = y, z = z,
        topology = topo,
        halo = halo)

    return new_grid
end

location_guide = Dict(:u => (Face, Center, Center),
                      :v => (Center, Face, Center),
                      :w => (Center, Center, Face))

function infer_location(field_name)
    if field_name in keys(location_guide)
        return location_guide[field_name]
    else
        return (Center, Center, Center)
    end
end

#####
##### set! for simulation models and observations
#####

"""
    column_ensemble_interior(batch::BatchedSyntheticObservations,
                             field_name, time_index, (Nensemble, Nbatch, Nz))

Return an `Nensemble × Nbatch × Nz` Array of `(1, 1, Nz)` `field_name` data,
given `Nbatch` `SyntheticObservations` objects. The `Nbatch × Nz` data for `field_name`
is copied `Nensemble` times to form a 3D Array.
"""
function column_ensemble_interior(batch::BatchedSyntheticObservations,
                                  field_name, time_index, (Nensemble, Nbatch, Nz))

    zeros_column = zeros(1, 1, Nz)
    Nt = length(first(batch).times)

    batched_data = []
    for observation in batch.observations
        fts = observation.field_time_serieses
        if field_name in keys(fts) && time_index <= Nt
            field_column = interior(fts[field_name][time_index])
            push!(batched_data, interior(fts[field_name][time_index]))
        else
            push!(batched_data, zeros_column)
        end
    end

    # Make a Vector of 1D Array into a 3D Array
    flattened_data = cat(batched_data..., dims = 2) # (Nbatch, Nz)
    ensemble_interior = cat((flattened_data for i = 1:Nensemble)..., dims = 1) # (Nensemble, Nbatch, Nz)

    return ensemble_interior
end

function set!(model, obs::SyntheticObservations, time_index=1)
    for field_name in keys(fields(model))
        model_field = fields(model)[field_name]

        if field_name ∈ keys(obs.field_time_serieses)
            obs_field = obs.field_time_serieses[field_name][time_index]
            set!(model_field, obs_field)
        elseif model_field isa Field
            fill!(parent(model_field), 0)
        end
    end

    update_state!(model)

    return nothing
end

function set!(model, observations::BatchedSyntheticObservations, time_index=1)
    for field_name in keys(fields(model))
        model_field = fields(model)[field_name]
        model_field_size = size(model_field)
        Nensemble = model.grid.Nx

        observations_data = column_ensemble_interior(observations, field_name, time_index, model_field_size)
    
        # Reshape `observations_data` to the size of `model_field`'s interior
        reshaped_data = arch_array(architecture(model_field), reshape(observations_data, size(model_field)))
    
        # Sets the interior of field `model_field` to values of `reshaped_data`
        model_field .= reshaped_data
    end

    update_state!(model)

    return nothing
end

#####
##### FieldTimeSeriesCollector for collecting data while a simulation runs
#####

struct FieldTimeSeriesCollector{G, D, F}
    grid :: G
    times :: Vector{Float64}
    field_time_serieses :: D
    collected_fields :: F
    collection_times :: Vector{Float64}
end

"""
    FieldTimeSeriesCollector(collected_fields, times;
                             architecture = CPU(),
                             averaging_window = nothing,
                             averaging_stride = nothing)

Return a `FieldTimeSeriesCollector` for `fields` of `simulation`.
`fields` is a `NamedTuple` of `AbstractField`s that are to be collected.
"""
function FieldTimeSeriesCollector(collected_fields, times;
                                  architecture = CPU(),
                                  averaging_window = nothing,
                                  averaging_stride = 1)

    grid = on_architecture(architecture, first(collected_fields).grid)
    field_time_serieses = Dict{Symbol, Any}()

    for field_name in keys(collected_fields)
        field = collected_fields[field_name]
        inds = indices(field)
        LX, LY, LZ = location(field)
        field_time_series = FieldTimeSeries{LX, LY, LZ}(grid, times; indices=inds)
        field_time_serieses[field_name] = field_time_series
    end

    # Convert to NamedTuple
    field_time_serieses = NamedTuple(name => field_time_serieses[name] for name in keys(collected_fields))

    collection_times = similar(times)

    # Wrap collected fields in WindowedTimeAverage if requested
    if !isnothing(averaging_window)
        schedule = AveragedSpecifiedTimes(times; window=averaging_window, stride=averaging_stride)
        wrap(field) = WindowedTimeAverage(field; schedule, fetch_operand=false)
        averaged_collected_fields = NamedTuple(name => wrap(collected_fields[name])
                                               for name in keys(collected_fields))
        collected_fields = averaged_collected_fields
    end

    return FieldTimeSeriesCollector(grid, times, field_time_serieses,
                                    collected_fields, collection_times)
end

# For using in a Callback
function (collector::FieldTimeSeriesCollector)(simulation)
    for field in collector.collected_fields
        compute!(field)
    end

    current_time = simulation.model.clock.time
    time_index = findfirst(t -> t ≈ current_time, collector.times)
    if isnothing(time_index)
        @warn string("Current time ", prettytime(current_time), " not found in
                     time collector times ", prettytime.(collector.times))
        return nothing
    end

    for field_name in keys(collector.collected_fields)
        field_time_series = collector.field_time_serieses[field_name]
        if architecture(collector.grid) != architecture(simulation.model.grid)
            arch = architecture(collector.grid)
            device_collected_field_data = arch_array(arch, parent(collector.collected_fields[field_name]))
            parent(field_time_series[time_index]) .= device_collected_field_data
        else
            set!(field_time_series[time_index], collector.collected_fields[field_name])
        end
    end

    # We _have_ collected data
    collector.collection_times[time_index] = current_time

    return nothing
end

#####
##### Initializing simulations
#####

nothingfunction(simulation) = nothing

function initialize_forward_run!(simulation,
                                 observations,
                                 time_series_collector,
                                 initialize_with_observations,
                                 initialize_simulation!,
                                 parameters)

    reset!(simulation)

    times = observation_times(observations)
    initial_time = times[1]
    simulation.model.clock.time = initial_time

    # Clear potential NaNs from timestepper data.
    # Particularly important for Adams-Bashforth timestepping scheme.
    # Oceananigans ≤ v0.71 initializes the Adams-Bashforth scheme with an Euler step by
    # *multiplying* the tendency at time-step n-1 by 0. Because 0 * NaN = NaN, this fails
    # when the tendency at n-1 contains NaNs.
    timestepper = simulation.model.timestepper 
    for field in tuple(timestepper.Gⁿ..., timestepper.G⁻...)
        if !isnothing(field)
            parent(field) .= 0
        end
    end
    
    # Initialize FieldTimeSeriesCollector
    time_series_collector.collection_times .= 0
    for time_series in time_series_collector.field_time_serieses
        parent(time_series) .= 0
    end

    # Add Callback for computing time-averages if necessary
    collection_schedule = SpecifiedTimes(times...)

    # Note that callbacks for computing time-averages must be added _before_ callbacks
    # for data collection
    for name in keys(time_series_collector.collected_fields)
        field = time_series_collector.collected_fields[name]
        if field isa WindowedTimeAverage
            # Replace averaging schedule
            field.schedule = AveragedSpecifiedTimes(collection_schedule;
                                                    window = field.schedule.window,
                                                    stride = field.schedule.stride)
            callback_name = Symbol(:time_average_, name)
            simulation.callbacks[callback_name] = Callback(field)
        end
    end

    :nan_checker ∈ keys(simulation.callbacks) && pop!(simulation.callbacks, :nan_checker)
    simulation.callbacks[:data_collector] = Callback(time_series_collector, collection_schedule)
    simulation.stop_time = times[end]

    if initialize_with_observations
        set!(simulation.model, observations, 1)
    end

    initialize_simulation!(simulation, parameters)

    return nothing
end

summarize_metadata(::Nothing) = ""
summarize_metadata(metadata) = keys(metadata)

function Base.show(io::IO, obs::SyntheticObservations)
    times_str = prettyvector(prettytime.(obs.times, false))

    print(io, "SyntheticObservations with fields $(propertynames(obs.field_time_serieses))", '\n',
              "├── times: $times_str", '\n',
              "├── grid: $(summary(obs.grid))", '\n',
              "├── path: \"$(obs.path)\"", '\n',
              "├── metadata: ", summarize_metadata(obs.metadata), '\n',
              "└── transformation: $(summary(obs.transformation))")
end

end # module

