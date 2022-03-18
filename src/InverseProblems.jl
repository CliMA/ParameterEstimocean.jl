module InverseProblems

export
    InverseProblem,
    forward_map,
    forward_run!,
    observation_map,
    observation_map_variance_across_time,
    ConcatenatedOutputMap

using OffsetArrays, Statistics, OrderedCollections
using Suppressor: @suppress

using ..Transformations: transform_field_time_series
using ..Parameters: new_closure_ensemble, transform_to_constrained

using ..Observations:
    AbstractObservation,
    SyntheticObservations,
    initialize_forward_run!,
    FieldTimeSeriesCollector,
    observation_times,
    forward_map_names

using Oceananigans: run!, fields, FieldTimeSeries, CPU
using Oceananigans.Architectures: architecture
using Oceananigans.OutputReaders: InMemory
using Oceananigans.Fields: interior, location
using Oceananigans.Grids: Flat, Bounded,
                          Face, Center,
                          RectilinearGrid, offset_data,
                          topology, halo_size,
                          interior_parent_indices

using Oceananigans.Models.HydrostaticFreeSurfaceModels: SingleColumnGrid, YZSliceGrid, ColumnEnsembleSize

import ..Observations: normalize!

#####
##### Output maps (maps from simulation output to observation space)
#####

abstract type AbstractOutputMap end

output_map_type(fp) = output_map_str(fp)

struct ConcatenatedOutputMap{T} <: AbstractOutputMap
    time_indices::T
end

ConcatenatedOutputMap(; time_indices = Colon()) = ConcatenatedOutputMap(time_indices)
    
output_map_str(::ConcatenatedOutputMap) = "ConcatenatedOutputMap"

"""
    ConcatenatedVectorNormMap()

Forward map transformation of simulation output to a scalar by
taking a naive `norm` of the difference between concatenated vectors of the
observations and simulation output.
"""
struct ConcatenatedVectorNormMap{T} <: AbstractOutputMap
    time_indices::T
end

ConcatenatedVectorNormMap(; time_indices = Colon()) = ConcatenatedVectorNormMap(time_indices)

output_map_str(::ConcatenatedVectorNormMap) = "ConcatenatedVectorNormMap"

initial_time_index(output_map::AbstractOutputMap) = output_map.time_indices == Colon() ? 
                                                    1 : first(output_map.time_indices)

#####
##### InverseProblems
#####

struct InverseProblem{F, O, S, T, P, I}
    observations :: O
    simulation :: S
    time_series_collector :: T
    free_parameters :: P
    output_map :: F
    initialize_simulation :: I
end

nothingfunction(simulation) = nothing

"""
    InverseProblem(observations,
                   simulation,
                   free_parameters;
                   output_map = ConcatenatedOutputMap(),
                   time_series_collector = nothing)

Return an `InverseProblem`.
"""
function InverseProblem(observations,
                        simulation,
                        free_parameters;
                        output_map = ConcatenatedOutputMap(),
                        time_series_collector = nothing,
                        initialize_simulation = nothingfunction)

    if isnothing(time_series_collector) # attempt to construct automagically
        simulation_fields = fields(simulation.model)
        collected_fields = NamedTuple(name => simulation_fields[name] for name in forward_map_names(observations))
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))
    end

    return InverseProblem(observations, simulation, time_series_collector, free_parameters, output_map, initialize_simulation)
end

Base.summary(ip::InverseProblem) =
    string("InverseProblem{", summary(ip.output_map), "} with free parameters ", ip.free_parameters.names)

function Base.show(io::IO, ip::InverseProblem)
    sim_str = "Simulation on $(summary(ip.simulation.model.grid)) with Δt=$(ip.simulation.Δt)"
    out_map_str = summary(ip.output_map)

    print(io, summary(ip), '\n',
        "├── observations: $(summary(ip.observations))", '\n',
        "├── simulation: $sim_str", '\n',
        "├── free_parameters: $(summary(ip.free_parameters))", '\n',
        "└── output map: $out_map_str")

    return nothing
end

tupify_parameters(ip, θ) = NamedTuple{ip.free_parameters.names}(Tuple(θ))
tupify_parameters(ip, θ::Union{Dict, NamedTuple}) = NamedTuple(name => θ[name] for name in ip.free_parameters.names)

"""
    expand_parameters(ip, θ)

Convert `θ` to `Vector{<:NamedTuple}`, where the elements
correspond to `ip.free_parameters`.

`θ` may represent an ensemble of parameter sets via:

* `θ::Vector{<:Vector}` (caution: parameters must be ordered correctly!)
* `θ::Matrix` (caution: parameters must be ordered correctly!)
* `θ::Vector{<:NamedTuple}` 

or a single parameter set if `θ::Vector{<:Number}`.

If `length(θ)` is less the the number of ensemble members in `ip.simulation`, the
last parameter set is copied to fill the parameter set ensemble.
"""
function expand_parameters(ip, θ::Vector)
    Nfewer = Nensemble(ip) - length(θ)
    Nfewer < 0 && throw(ArgumentError("There are $(-Nfewer) more parameter sets than ensemble members!"))

    θ = [tupify_parameters(ip, θi) for θi in θ]

    # Fill out parameter set ensemble
    Nfewer > 0 && append!(θ, [θ[end] for _ = 1:Nfewer])

    return θ
end

# Expand single parameter set
expand_parameters(ip, θ::Union{NamedTuple, Vector{<:Number}}) = expand_parameters(ip, [θ])

# Convert matrix to vector of vectors
expand_parameters(ip, θ::Matrix) = expand_parameters(ip, [θ[:, k] for k = 1:size(θ, 2)])

#####
##### Forward map evaluation given vector-of-vector (one parameter vector for each ensemble member)
#####

const OneDimensionalEnsembleGrid = RectilinearGrid{<:Any, Flat, Flat, Bounded}
const TwoDimensionalEnsembleGrid = RectilinearGrid{<:Any, Flat, Bounded, Bounded}

Nobservations(grid::OneDimensionalEnsembleGrid) = grid.Ny
Nobservations(grid::TwoDimensionalEnsembleGrid) = 1

Nensemble(grid::Union{OneDimensionalEnsembleGrid, TwoDimensionalEnsembleGrid}) = grid.Nx
Nensemble(ip::InverseProblem) = Nensemble(ip.simulation.model.grid)

""" 
    observation_map(ip::InverseProblem)

Transform and return `ip.observations` appropriate for `ip.output_map`. 
"""
observation_map(ip::InverseProblem) = observation_map(ip.output_map, ip.observations)

observation_map(map::ConcatenatedOutputMap, observations) = transform_time_series(map, observations)
observation_map(map::ConcatenatedVectorNormMap, observations) = hcat(0.0)

"""
    forward_run!(ip, parameters)

Initialize `ip.simulation` with `parameters` and run it forward. Output is stored
in `ip.time_series_collector`.
"""
function forward_run!(ip::InverseProblem, parameters; suppress=false)
    observations = ip.observations
    simulation = ip.simulation
    closures = simulation.model.closure

    θ = expand_parameters(ip, parameters)
    simulation.model.closure = new_closure_ensemble(closures, θ, architecture(simulation.model.grid))

    initialize_forward_run!(simulation, observations, ip.time_series_collector, ip.initialize_simulation)

    if suppress
        @suppress run!(simulation)
    else
        run!(simulation)
    end
    
    return nothing
end

"""
    forward_map(ip, parameters)

Run `ip.simulation` forward with `parameters` and return the data,
transformed into an array format expected by `EnsembleKalmanProcesses.jl`.
"""
function forward_map(ip, parameters; suppress=true)

    # Run the simulation forward and populate the time series collector
    # with model data.
    forward_run!(ip, parameters; suppress)

    # Verify that data was collected properly
    all(ip.time_series_collector.times .≈ ip.time_series_collector.collection_times) ||
        error("FieldTimeSeriesCollector.collection_times does not match FieldTimeSeriesCollector.times. \n" *
              "Field time series data may not have been properly collected")

    # Transform the model data according to `ip.output_map` into
    # the array format expected by EnsembleKalmanProcesses.jl
    # The result has `size(output) = (output_size, ensemble_capacity)`,
    # where `output_size` is determined by both the `output_map` and the
    # data collection dictated by `ip.observations`.
    output = transform_forward_map_output(ip.output_map, ip.observations, ip.time_series_collector)

    # (Nobservations, Nensemble)
    return output
end

(ip::InverseProblem)(θ) = forward_map(ip, θ)

"""
    inverting_forward_map(ip::InverseProblem, X)

Transform unconstrained parameters `X` into constrained,
physical-space parameters `θ` and execute `forward_map(ip, θ)`.
"""
function inverting_forward_map(ip::InverseProblem, X)
    θ = transform_to_constrained(ip.free_parameters.priors, X)
    return forward_map(ip, θ)
end

#####
##### ConcatenatedOutputMap
#####

# Need docstrings
struct ConcatenatedOutputMap end

# for field_name in forward_map_names(observation)
#     field_time_series = observation.field_time_serieses[field_name]
#     field_time_series_interior = Array(interior(field_time_series))

#     # Ignore initial condition given by first element in map.time_indices
#     field_time_series_data = output_map.time_indices == Colon() ? 
#         selectdim(field_time_series_interior, 4, 2:size(field_time_series_interior, 4)) :
#         selectdim(field_time_series_interior, 4, output_map.time_indices[2:end])

"""
    transform_time_series(::ConcatenatedOutputMap, time_series::SyntheticObservations)
Concatenates flattened, normalized data for each field in the `time_series`.
"""
function transform_time_series(output_map::ConcatenatedOutputMap, time_series::SyntheticObservations)
    flattened_normalized_data = []

    for field_name in keys(time_series.field_time_serieses)
        field_time_series = time_series.field_time_serieses[field_name]
        field_time_series_interior = Array(interior(field_time_series))

        # Ignore initial condition given by first element in map.time_indices
        field_time_series_data = output_map.time_indices == Colon() ? 
            selectdim(field_time_series_interior, 4, 2:size(field_time_series_interior, 4)) :
            selectdim(field_time_series_interior, 4, output_map.time_indices[2:end])

        # Normalize data according to observation-specified normalization
        normalize!(field_time_series_data, time_series.normalization[field_name])

        # Reshape data to 2D array with size (Nx, :)
        Nx, Ny, Nz, Nt = size(field_time_series_data)
        field_time_series_data = reshape(field_time_series_data, Nx, Ny * Nz * Nt)

        push!(flattened_normalized_data, field_time_series_data)
    end

    transformed = hcat(flattened_normalized_data...)

    return Matrix(transpose(transformed))
end

"""
    transform_time_series(::ConcatenatedOutputMap, observation::SyntheticObservations)

Transforms, normalizes, and concatenates data for field time series in `observation`.
"""
function transform_time_series(::ConcatenatedOutputMap, observation::SyntheticObservations)
    data_vector = []

    for field_name in forward_map_names(observation)
        # Transform time series data observation-specified `transformation`
        field_time_series = observation.field_time_serieses[field_name]
        transformation = observation.transformation[field_name]
        transformed_datum = transform_field_time_series(transformation, field_time_series)

        # Build out array
        push!(data_vector, transformed_datum)
    end

    # Concatenate!
    concatenated_data = hcat(data_vector...)

    return Matrix(transpose(concatenated_data))
end

"""
    transform_time_series(map, batched_observations::Vector)

Concatenate the output of `transform_time_series` of each observation
in `batched_observations`.
"""
transform_time_series(map, batched_observations::Vector) =
    vcat(Tuple(transform_time_series(map, obs) for obs in batched_observations)...)

const BatchedOrSingletonObservations = Union{SyntheticObservations,
                                             Vector{<:SyntheticObservations}}

function transform_forward_map_output(map::ConcatenatedOutputMap,
                                      observations::BatchedOrSingletonObservations,
                                      time_series_collector)

    # transposed_output isa Vector{SyntheticObservations} where SyntheticObservations is Nx by Nz by Nt
    transposed_forward_map_output = transpose_model_output(time_series_collector, observations)

    return transform_time_series(map, transposed_forward_map_output)
end

function transform_output(output_map::ConcatenatedVectorNormMap,
    observations::Union{SyntheticObservations, Vector{<:SyntheticObservations}},
    time_series_collector)

    concat_map = ConcatenatedOutputMap(output_map.time_indices)
    fwd_map = transform_output(concat_map, observations, time_series_collector)
    obs_map = transform_time_series(concat_map, observations)

    diffn = fwd_map .- obs_map
    return sqrt.(mapslices(norm, diffn; dims = 1))
end

vectorize(observation) = [observation]
vectorize(observations::Vector) = observations

# Dispatch transpose_model_output based on collector grid
transpose_model_output(time_series_collector, observations) =
    transpose_model_output(time_series_collector.grid, time_series_collector, observations)

transpose_model_output(collector_grid::YZSliceGrid, time_series_collector, observations) =
    SyntheticObservations(time_series_collector.field_time_serieses,
                          observations.forward_map_names,
                          collector_grid,
                          time_series_collector.times,
                          nothing,
                          nothing,
                          observations.transformation)

"""
    transpose_model_output(time_series_collector, observations)

Transpose a `NamedTuple` of 4D `FieldTimeSeries` model output collected by `time_series_collector`
into a Vector of `SyntheticObservations` for each member of the observation batch.

Return a 1-vector in the case of singleton observations.
"""
function transpose_model_output(collector_grid::SingleColumnGrid, time_series_collector, observations)
    observations = vectorize(observations)
    times = time_series_collector.times

    transposed_output = []

    Nensemble = collector_grid.Nx
    Nbatch =  collector_grid.Ny
    Nz = collector_grid.Nz
    Hz = collector_grid.Hz
    Nt = length(times)

    grid = drop_y_dimension(collector_grid)

    for j = 1:Nbatch
        observation = observations[j]
        time_serieses = OrderedDict{Any, Any}()

        for name in forward_map_names(observation)
            loc = LX, LY, LZ = location(observation.field_time_serieses[name])
            topo = topology(grid)

            field_time_series = time_series_collector.field_time_serieses[name]

            raw_data = parent(field_time_series.data)
            data = OffsetArray(view(raw_data, :, j:j, :, :), 0, 0, -Hz, 0)

            time_series = FieldTimeSeries{LX, LY, LZ, InMemory}(data, grid, nothing, times)
            time_serieses[name] = time_series
        end

        # Convert to NamedTuple
        time_serieses = NamedTuple(name => time_series for (name, time_series) in time_serieses)

        batch_output = SyntheticObservations(time_serieses,
                                             observation.forward_map_names,   
                                             grid,
                                             times,
                                             nothing,
                                             nothing,
                                             observation.transformation)

        push!(transposed_output, batch_output)
    end

    return transposed_output
end

function drop_y_dimension(grid::SingleColumnGrid)
    new_size = ColumnEnsembleSize(Nz=grid.Nz, ensemble=(grid.Nx, 1), Hz=grid.Hz)
    new_halo_size = ColumnEnsembleSize(Nz=1, Hz=grid.Hz)
    z_domain = (grid.zᵃᵃᶠ[1], grid.zᵃᵃᶠ[grid.Nz])
    new_grid = RectilinearGrid(size=new_size, halo=new_halo_size, z=z_domain, topology=(Flat, Flat, Bounded))
    return new_grid
end

#####
##### VectorNormMap
#####

struct VectorNormMap end

observation_map(::VectorNormMap, observations) = reshape([0], 1, 1)

function transform_forward_map_output(::VectorNormMap, obs, time_series_collector)
    # Collected concatenated output and observations
    G = transform_forward_map_output(ConcatenatedOutputMap(), obs, time_series_collector)
    y = observation_map(ConcatenatedOutputMap(), obs)

    # Compute vector norm across ensemble members. result should be
    # (1, Nensemble)
    return mapslices(Gᵏ -> norm(Gᵏ - y), G, dims=1)
end

#####
##### Utils
#####

"""
    observation_map_variance_across_time(map::ConcatenatedOutputMap, observation::SyntheticObservations)

Return an array of size `(Nensemble, Ny * Nz * Nfields, Ny * Nz * Nfields)` that stores the covariance of
each element of the observation map measured across time, for each ensemble member, where `Nensemble` is
the ensemble size, `Ny` is either the number of grid elements in `y` or the batch size, `Nz` is the number
of grid elements in the vertical, and `Nfields` is the number of fields in `observation`.
"""
function observation_map_variance_across_time(map::ConcatenatedOutputMap, observation::SyntheticObservations)
    # These aren't right because every field can have a different transformation, so...
    Nx, Ny, Nz = size(observation.grid)
    Nt = length(first(observation.transformation).time)

    Nfields = length(forward_map_names(observation))

    y = transform_time_series(map, observation)
    @assert length(y) == Nx * Ny * Nz * Nt * Nfields # otherwise we're headed for trouble...

    y = transpose(y) # (Nx, Ny*Nz*Nt*Nfields)

    # Transpose `Nfields` dimension
    reshaped_y = reshape(y, Nx, Ny * Nz, Nt, Nfields) # (Nx, Ny*Nz, Nt, Nfields)
    permuted_y = permutedims(y, [1, 2, 4, 3])
    reshaped_permuted_y = reshape(permuted_y, Nx, Ny * Nz * Nfields, Nt)

    # Compute `var`iance across time
    dataset = [reshape(var(reshaped_permuted_y[:, :, 1:n], dims = 3), Nx, Ny * Nz, Nfields) for n = 1:Nt]
    concatenated_dataset = cat(dataset..., dims = 2)
    replace!(concatenated_dataset, NaN => 0) # variance for first time step is zero

    return reshape(concatenated_dataset, Nx, Ny * Nz * Nt * Nfields)
end

observation_map_variance_across_time(map::ConcatenatedOutputMap, observations::Vector) =
    hcat(Tuple(observation_map_variance_across_time(map, observation) for observation in observations)...)

observation_map_variance_across_time(ip::InverseProblem) = observation_map_variance_across_time(ip.output_map, ip.observations)

end # module
