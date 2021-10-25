module InverseProblems

using OrderedCollections

using ..Observations: obs_str, AbstractObservation, OneDimensionalTimeSeries, initialize_simulation!, FieldTimeSeriesCollector, 
                      observation_times

using ..TurbulenceClosureParameters: free_parameters_str, update_closure_ensemble_member!

using OffsetArrays

using Oceananigans: short_show, run!, fields, FieldTimeSeries, CPU
using Oceananigans.OutputReaders: InMemory
using Oceananigans.Fields: interior, location
using Oceananigans.Grids: Flat, Bounded,
                          Face, Center,
                          RegularRectilinearGrid, offset_data,
                          topology, halo_size,
                          interior_parent_indices

import ..Observations: normalize!

#####
##### Output maps (maps from simulation output to observation space)
#####

abstract type AbstractOutputMap end

output_map_type(fp) = output_map_str(fp)

struct ConcatenatedOutputMap end

output_map_str(::ConcatenatedOutputMap) = "ConcatenatedOutputMap"

"""
    ConcatenatedVectorNormMap()

Forward map transformation of simulation output to a scalar by
taking a naive `norm` of the difference between concatenated vectors of the
observations and simulation output.
"""
struct ConcatenatedVectorNormMap end

output_map_str(::ConcatenatedVectorNormMap) = "ConcatenatedVectorNormMap"

#####
##### InverseProblems
#####

struct InverseProblem{F, O, S, T, P}
    observations :: O
    simulation :: S
    time_series_collector :: T
    free_parameters :: P
    output_map :: F
end

"""
    InverseProblem(observations, simulation, free_parameters; output_map=ConcatenatedOutputMap())

Return an `InverseProblem`.
"""
function InverseProblem(observations, simulation, free_parameters; output_map=ConcatenatedOutputMap(), time_series_collector=nothing)

    if isnothing(time_series_collector) # attempt to construct automagically
        simulation_fields = fields(simulation.model)
        collected_fields = NamedTuple(name => simulation_fields[name] for name in keys(simulation_fields))
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))
    end

    return InverseProblem(observations, simulation, time_series_collector, free_parameters, output_map)
end

function Base.show(io::IO, ip::InverseProblem)
    sim_str = "Simulation on $(short_show(ip.simulation.model.grid)) with Δt=$(ip.simulation.Δt)"

    out_map_type = output_map_type(ip.output_map)
    out_map_str = output_map_str(ip.output_map)

    print(io, "InverseProblem{$out_map_type}", '\n',
              "├── observations: $(obs_str(ip.observations))", '\n',    
              "├── simulation: $sim_str", '\n',
              "├── free_parameters: $(free_parameters_str(ip.free_parameters))", '\n',
              "└── output map: $out_map_str")

    return nothing
end

tupify_parameters(ip, θ) = NamedTuple{ip.free_parameters.names}(Tuple(θ))

#####
##### Forward map evaluation given vector-of-vector (one parameter vector for each ensemble member)
#####

const OneDimensionalEnsembleGrid = RegularRectilinearGrid{<:Any, Flat, Flat, Bounded}

n_ensemble(grid::OneDimensionalEnsembleGrid) = grid.Nx
n_observations(grid::OneDimensionalEnsembleGrid) = grid.Ny
n_z(grid::OneDimensionalEnsembleGrid) = grid.Nz

n_ensemble(ip::InverseProblem) = n_ensemble(ip.simulation.model.grid)

""" Transform and return `ip.observations` appropriate for `ip.output_map`. """
observation_map(ip::InverseProblem) = transform_observations(ip.output_map, ip.observations)

function forward_map(ip::InverseProblem, θ::Vector{<:NamedTuple})
    observations = ip.observations
    simulation = ip.simulation
    closures = simulation.model.closure

    for p in 1:length(θ)
        update_closure_ensemble_member!(closures, p, θ[p])
    end

    initialize_simulation!(simulation, observations, ip.time_series_collector)
    run!(simulation)

    return transform_output(ip.output_map, observations, ip.time_series_collector)
end

forward_map(ip, θ::Vector{<:Vector}) = forward_map(ip, [tupify_parameters(ip, p) for p in θ])
forward_map(ip, θ::Matrix) = forward_map(ip, [tupify_parameters(ip, θ[:, i]) for i in 1:size(θ, 2)])

(ip::InverseProblem)(θ) = forward_map(ip, θ)

function transform_observations(::ConcatenatedOutputMap, observation::OneDimensionalTimeSeries)
    flattened_normalized_data = []

    for field_name in keys(observation.field_time_serieses)
        field_time_series = observation.field_time_serieses[field_name]

        # *** FIXME ***
        # Here we hack an implementation of `interior` because
        # `field_time_series.grid` may be wrong (sometimes grid.Nx is wrong)
        grid = field_time_series.grid
        Hx, Hy, Hz = halo_size(grid)
        Nx, Ny, Nz = size(grid)
        topo = topology(grid)
        loc = location(field_time_series)
        y_indices = interior_parent_indices(loc[2], topo[2], Ny, Hy)
        z_indices = interior_parent_indices(loc[3], topo[3], Nz, Hz)
        field_time_series_data = Array(view(parent(field_time_series), :, y_indices, z_indices, :))

        Nx, Ny, Nz, Nt = size(field_time_series_data)
        field_time_series_data = reshape(field_time_series_data, Nx, Ny * Nz * Nt)

        normalize!(field_time_series_data, observation.normalization[field_name])

        push!(flattened_normalized_data, field_time_series_data)
    end

    transformed = hcat(flattened_normalized_data...)

    return Matrix(transpose(transformed))
end

transform_observations(map, observations::Vector) =
    hcat(Tuple(transform_observations(map, observation) for observation in observations)...)

function transform_output(map::ConcatenatedOutputMap,
                          observations::Union{OneDimensionalTimeSeries, Vector{<:OneDimensionalTimeSeries}},
                          time_series_collector)

    # transposed_output isa Vector{OneDimensionalTimeSeries} where OneDimensionalTimeSeries is Nx by Nz by Nt
    transposed_output = transpose_model_output(time_series_collector, observations)[1]

    return transform_observations(map, transposed_output)
end

vectorize(observation) = [observation]
vectorize(observations::Vector) = observations

"""
    transpose_model_output(time_series_collector, observations)

Transpose a `NamedTuple` of 4D `FieldTimeSeries` model output collected by `time_series_collector`
into a Vector of `OneDimensionalTimeSeries` for each member of the observation batch.

Return a 1-vector in the case of singleton observations.
"""
function transpose_model_output(time_series_collector, observations)
    observations = vectorize(observations)
    times = time_series_collector.times

    transposed_output = []

    n_ensemble = time_series_collector.grid.Nx
    n_batch = time_series_collector.grid.Ny
    Nz = time_series_collector.grid.Nz
    Hz = time_series_collector.grid.Hz
    Nt = length(times)

    for j = 1:n_batch
        observation = observations[j]
        grid = observation.grid
        time_serieses = OrderedDict{Any, Any}()

        for name in keys(observation.field_time_serieses)
            loc = LX, LY, LZ = location(observation.field_time_serieses[name])
            topo = topology(grid)

            field_time_series = time_series_collector.field_time_serieses[name]

            raw_data = parent(field_time_series.data)
            data = OffsetArray(view(raw_data, :, j:j, :, :), 0, 0, -Hz, 0)

            # Note: FieldTimeSeries.grid.Nx is in general incorrect
            time_series = FieldTimeSeries{LX, LY, LZ, InMemory}(data, CPU(), grid, nothing, times)
            time_serieses[name] = time_series
        end

        # Convert to NamedTuple
        time_serieses = NamedTuple(name => time_series for (name, time_series) in time_serieses)


        batch_output = OneDimensionalTimeSeries(time_serieses,
                                                grid, # this grid has the wrong grid.Nx --- forgive us our sins
                                                times,
                                                nothing,
                                                nothing,
                                                observation.normalization) 

        push!(transposed_output, batch_output)
    end

    return transposed_output
end

end # module