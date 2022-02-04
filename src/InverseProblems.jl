module InverseProblems

using OrderedCollections
using Suppressor: @suppress

using ..Observations: AbstractObservation, SyntheticObservations, initialize_simulation!, FieldTimeSeriesCollector,
    observation_times, observation_names

using ..Parameters: new_closure_ensemble

using OffsetArrays, Statistics, LinearAlgebra

using Oceananigans: run!, fields, FieldTimeSeries, CPU
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
function InverseProblem(observations,
                        simulation,
                        free_parameters;
                        output_map = ConcatenatedOutputMap(),
                        time_series_collector = nothing)

    if isnothing(time_series_collector) # attempt to construct automagically
        simulation_fields = fields(simulation.model)
        collected_fields = NamedTuple(name => simulation_fields[name] for name in observation_names(observations))
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations))
    end

    return InverseProblem(observations, simulation, time_series_collector, free_parameters, output_map)
end

function Base.show(io::IO, ip::InverseProblem)
    sim_str = "Simulation on $(summary(ip.simulation.model.grid)) with Δt=$(ip.simulation.Δt)"

    out_map_type = output_map_type(ip.output_map)
    out_map_str = output_map_str(ip.output_map)

    print(io, "InverseProblem{$out_map_type}", '\n',
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

""" Transform and return `ip.observations` appropriate for `ip.output_map`. """
observation_map(ip::InverseProblem) = observation_map(ip.output_map, ip.observations)
observation_map(map::ConcatenatedOutputMap, observations) = transform_time_series(map, observations)
observation_map(map::ConcatenatedVectorNormMap, observations) = hcat(0.0)

"""
    forward_run!(ip, parameters)

Initialize `ip.simulation` with `parameters` and run it forward. Output is stored
in `ip.time_series_collector`.
"""
function forward_run!(ip::InverseProblem, parameters)
    observations = ip.observations
    simulation = ip.simulation
    closures = simulation.model.closure

    θ = expand_parameters(ip, parameters)
    simulation.model.closure = new_closure_ensemble(closures, θ)

    initialize_simulation!(simulation, observations, ip.time_series_collector, initial_time_index(ip.output_map))

    @suppress run!(simulation)
    
    return nothing
end

"""
    forward_map(ip, parameters)

Run `ip.simulation` forward with `parameters` and return the data,
transformed into an array format expected by `EnsembleKalmanProcesses.jl`.
"""
function forward_map(ip, parameters)

    # Run the simulation forward and populate the time series collector
    # with model data.
    forward_run!(ip, parameters)

    # Transform the model data according to `ip.output_map` into
    # the array format expected by EnsembleKalmanProcesses.jl
    # The result has `size(output) = (output_size, ensemble_capacity)`,
    # where `output_size` is determined by both the `output_map` and the
    # data collection dictated by `ip.observations`.
    output = transform_output(ip.output_map, ip.observations, ip.time_series_collector)

    # (output_size, ensemble_size)
    return output
end

(ip::InverseProblem)(θ) = forward_map(ip, θ)

"""
    transform_time_series(::ConcatenatedOutputMap, time_series::SyntheticObservations)

Concatenates flattened, normalized data for each field in the `time_series`.
"""
function transform_time_series(output_map::ConcatenatedOutputMap, time_series::SyntheticObservations)
    flattened_normalized_data = []

    for field_name in keys(time_series.field_time_serieses)
        field_time_series = time_series.field_time_serieses[field_name]
        A = Array(interior(field_time_series))

        # Ignore initial condition given by first element in map.time_indices
        field_time_series_data = output_map.time_indices == Colon() ? 
            selectdim(A, 4, 2:size(A,4)) :
            selectdim(A, 4, output_map.time_indices[2:end])

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
    transform_time_series(map, time_serieses::Vector)

Return the `transform_time_series` of each `time_series` in `time_serieses` vector.
"""
transform_time_series(map, time_serieses::Vector) =
    vcat(Tuple(transform_time_series(map, time_series) for time_series in time_serieses)...)    

function transform_output(map::ConcatenatedOutputMap,
                          observations::Union{SyntheticObservations, Vector{<:SyntheticObservations}},
                          time_series_collector)

    # transposed_output isa Vector{SyntheticObservations} where SyntheticObservations is Nx by Nz by Nt
    transposed_output = transpose_model_output(time_series_collector, observations)

    return transform_time_series(map, transposed_output)
end

function transform_output(output_map::ConcatenatedVectorNormMap,
    observations::Union{SyntheticObservations,Vector{<:SyntheticObservations}},
    time_series_collector)

    concat_map = ConcatenatedOutputMap(output_map.time_indices)
    fwd_map = transform_output(concat_map, observations, time_series_collector)
    obs_map = transform_time_series(concat_map, observations)

    diffn = fwd_map .- obs_map
    return sqrt.(mapslices(norm, diffn; dims = 1))
end

vectorize(observation) = [observation]
vectorize(observations::Vector) = observations

const YZSliceObservations = SyntheticObservations{<:Any, <:YZSliceGrid}

transpose_model_output(time_series_collector, observations::YZSliceObservations) =
    SyntheticObservations(time_series_collector.field_time_serieses,
                          time_series_collector.grid,
                          time_series_collector.times,
                          nothing,
                          nothing,
                          observations.normalization)

"""
    transpose_model_output(time_series_collector, observations)

Transpose a `NamedTuple` of 4D `FieldTimeSeries` model output collected by `time_series_collector`
into a Vector of `SyntheticObservations` for each member of the observation batch.

Return a 1-vector in the case of singleton observations.
"""
function transpose_model_output(time_series_collector, observations)
    observations = vectorize(observations)
    times = time_series_collector.times

    transposed_output = []

    Nensemble = time_series_collector.grid.Nx
    Nbatch = time_series_collector.grid.Ny
    Nz = time_series_collector.grid.Nz
    Hz = time_series_collector.grid.Hz
    Nt = length(times)

    grid = drop_y_dimension(time_series_collector.grid)

    for j = 1:Nbatch
        observation = observations[j]
        time_serieses = OrderedDict{Any, Any}()

        for name in keys(observation.field_time_serieses)
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
                                             grid,
                                             times,
                                             nothing,
                                             nothing,
                                             observation.normalization)

        push!(transposed_output, batch_output)
    end

    return transposed_output
end

function drop_y_dimension(grid::RectilinearGrid{<:Any, <:Flat, <:Flat, <:Bounded})
    new_size = ColumnEnsembleSize(Nz=grid.Nz, ensemble=(grid.Nx, 1), Hz=grid.Hz)
    new_halo_size = ColumnEnsembleSize(Nz=1, Hz=grid.Hz)
    z_domain = (grid.zᵃᵃᶠ[1], grid.zᵃᵃᶠ[grid.Nz])
    new_grid = RectilinearGrid(size=new_size, halo=new_halo_size, z=z_domain, topology=(Flat, Flat, Bounded))
    return new_grid
end

"""
    observation_map_variance_across_time(map::ConcatenatedOutputMap, observation::SyntheticObservations)

Returns an (Nx, Ny*Nz*Nfields, Ny*Nz*Nfields) array storing the covariance of each element of the observation 
map measured across time, for each ensemble member, where `Nx` is the ensemble size, `Ny` is the batch size, 
`Nz` is the number of grid elements in the vertical, and `Nfields` is the number of fields in `observation`.
"""
function observation_map_variance_across_time(map::ConcatenatedOutputMap, observation::SyntheticObservations)

    N_fields = length(keys(observation.field_time_serieses))

    a = transform_time_series(map, observation)
    a = transpose(a) # (Nx, Ny*Nz*Nt*Nfields)

    example_field_time_series = values(observation.field_time_serieses)[1]

    Nx, Ny, Nz, Nt = size(interior(example_field_time_series))

    # Assume all fields have the same size
    b = reshape(a, Nx, Ny * Nz, Nt, N_fields) # (Nx, Ny*Nz, Nt, Nfields)

    c = cat((b[:, :, :, i] for i = 1:N_fields)..., dims = 2) # (Nx, Ny*Nz*Nfields, Nt)

    ds = [reshape(var(c[:, :, 1:t], dims = 3), Nx, Ny * Nz, N_fields) for t = 1:Nt]

    e = cat(ds..., dims = 2)

    replace!(e, NaN => 0) # variance for first time step is zero

    return reshape(e, Nx, Ny * Nz * Nt * N_fields)
end

observation_map_variance_across_time(map::ConcatenatedOutputMap, observations::Vector) =
    hcat(Tuple(observation_map_variance_across_time(map, observation) for observation in observations)...)

observation_map_variance_across_time(ip::InverseProblem) = observation_map_variance_across_time(ip.output_map, ip.observations)

end # module

