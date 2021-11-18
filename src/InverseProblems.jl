module InverseProblems

using OrderedCollections

using ..Observations: obs_str, AbstractObservation, OneDimensionalTimeSeries, initialize_simulation!, FieldTimeSeriesCollector, 
                      observation_times, observation_names

using ..TurbulenceClosureParameters: free_parameters_str, update_closure_ensemble_member!

using OffsetArrays, Statistics

using Oceananigans: short_show, run!, fields, FieldTimeSeries, CPU
using Oceananigans.OutputReaders: InMemory
using Oceananigans.Fields: interior, location
using Oceananigans.Grids: Flat, Bounded,
                          Face, Center,
                          RegularRectilinearGrid, offset_data,
                          topology, halo_size,
                          interior_parent_indices

using Oceananigans.Models.HydrostaticFreeSurfaceModels: SingleColumnGrid, YZSliceGrid, ColumnEnsembleSize

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
        collected_fields = NamedTuple(name => simulation_fields[name] for name in observation_names(observations))
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

"""
    expand_parameters(ip, θ)

Convert `θ` to `Vector{<:NamedTuple}`, where the elements
correspond to `ip.free_parameters`.

`θ` may be `Vector{<:Vector}` or `Matrix` (correpsonding to a parameter
ensemble), or `Vector{<:Number}` (correpsonding to a single parameter vector).
"""
function expand_parameters(ip, θ::Vector{<:Vector})
    ensemble_capacity = n_ensemble(ip.time_series_collector.grid)
    ensemble_size = length(θ)
    
    θs = [tupify_parameters(ip, p) for p in θ]

    # feed redundant parameters in case ensemble_size < ensemble_capacity
    full_θs = vcat(θs , [θs[end] for _ in 1:(ensemble_capacity - ensemble_size)])

    return full_θs
end

expand_parameters(ip, θ::Vector{<:Number}) = expand_parameters(ip, [θ,])
expand_parameters(ip, θ::Matrix) = expand_parameters(ip, [θ[:, i] for i in 1:size(θ, 2)])

#####
##### Forward map evaluation given vector-of-vector (one parameter vector for each ensemble member)
#####

const OneDimensionalEnsembleGrid = RegularRectilinearGrid{<:Any, Flat, Flat, Bounded}
const TwoDimensionalEnsembleGrid = RegularRectilinearGrid{<:Any, Flat, Bounded, Bounded}

n_ensemble(grid::Union{OneDimensionalEnsembleGrid, TwoDimensionalEnsembleGrid}) = grid.Nx
n_observations(grid::OneDimensionalEnsembleGrid) = grid.Ny
n_observations(grid::TwoDimensionalEnsembleGrid) = 1
n_z(grid::Union{OneDimensionalEnsembleGrid, TwoDimensionalEnsembleGrid}) = grid.Nz
n_y(grid::TwoDimensionalEnsembleGrid) = grid.Ny
n_ensemble(ip::InverseProblem) = n_ensemble(ip.simulation.model.grid)

""" Transform and return `ip.observations` appropriate for `ip.output_map`. """
observation_map(ip::InverseProblem) = transform_observations(ip.output_map, ip.observations)

"""
    forward_run!(ip, parameters)

Initialize `ip.simulation` with `parameters` and run it forward.
Output will be stored in `ip.time_series_collector`.
"""
function forward_run!(ip::InverseProblem, parameters)
    observations = ip.observations
    simulation = ip.simulation
    closures = simulation.model.closure

    θ = expand_parameters(ip, parameters)

    for p in 1:length(θ)
        update_closure_ensemble_member!(closures, p, θ[p])
    end

    initialize_simulation!(simulation, observations, ip.time_series_collector)

    run!(simulation)
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

function transform_observations(::ConcatenatedOutputMap, observation::OneDimensionalTimeSeries)
    flattened_normalized_data = []

    for field_name in keys(observation.field_time_serieses)
        field_time_series = observation.field_time_serieses[field_name]
        field_time_series_data = Array(interior(field_time_series))

        Nx, Ny, Nz, Nt = size(field_time_series_data)
        field_time_series_data = reshape(field_time_series_data, Nx, Ny * Nz * Nt)

        normalize!(field_time_series_data, observation.normalization[field_name])

        push!(flattened_normalized_data, field_time_series_data)
    end

    transformed = hcat(flattened_normalized_data...)

    return Matrix(transpose(transformed))
end

transform_observations(map, observations::Vector) =
    vcat(Tuple(transform_observations(map, observation) for observation in observations)...)

"""
    observation_map_variance_across_time(map::ConcatenatedOutputMap, observation::OneDimensionalTimeSeries)

Returns an (Nx, Ny*Nz*Nfields, Ny*Nz*Nfields) array storing the covariance of each element of the observation 
map measured across time, for each ensemble member, where `Nx` is the ensemble size, `Ny` is the batch size, 
`Nz` is the number of grid elements in the vertical, and `Nfields` is the number of fields in `observation`.
"""
function observation_map_variance_across_time(map::ConcatenatedOutputMap, observation::OneDimensionalTimeSeries)

    N_fields = length(keys(observation.field_time_serieses))

    a = transform_observations(map, observation)
    a = transpose(a) # (Nx, Ny*Nz*Nt)

    example_field_time_series = values(observation.field_time_serieses)[1]

    Nx, Ny, Nz, Nt = size(interior(example_field_time_series))

    # Assume all fields have the same size
    b = reshape(a, Nx, Ny * Nz, Nt, N_fields); # (Nx, Ny*Nz, Nt, Nfields)

    c = cat((b[:, :, :, i] for i in 1:N_fields)..., dims=2) # (Nx, Ny*Nz*Nfields, Nt)

    ds = [reshape(var(c[:, :, 1:t], dims=3), Nx, Ny * Nz, N_fields) for t in 1:Nt]

    e = cat(ds..., dims=2)

    replace!(e, NaN => 0) # variance for first time step is zero

    return reshape(e, Nx, Ny*Nz*Nt*N_fields)
end

observation_map_variance_across_time(map::ConcatenatedOutputMap, observations::Vector) = 
    hcat(Tuple(observation_map_variance_across_time(map, observation) for observation in observations)...)

observation_map_variance_across_time(ip::InverseProblem) = observation_map_variance_across_time(ip.output_map, ip.observations)

function transform_output(map::ConcatenatedOutputMap,
                          observations::Union{OneDimensionalTimeSeries, Vector{<:OneDimensionalTimeSeries}},
                          time_series_collector)

    # transposed_output isa Vector{OneDimensionalTimeSeries} where OneDimensionalTimeSeries is Nx by Nz by Nt
    transposed_output = transpose_model_output(time_series_collector, observations)

    return transform_observations(map, transposed_output)
end

vectorize(observation) = [observation]
vectorize(observations::Vector) = observations

const YZSliceObservations = OneDimensionalTimeSeries{<:Any, <:YZSliceGrid}

function transpose_model_output(time_series_collector, observations::YZSliceObservations)
    return OneDimensionalTimeSeries(time_series_collector.field_time_serieses,
                                    time_series_collector.grid,
                                    time_series_collector.times,
                                    nothing,
                                    nothing,
                                    observations.normalization)
end

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

    grid = drop_y_dimension(time_series_collector.grid)

    for j = 1:n_batch
        observation = observations[j]
        time_serieses = OrderedDict{Any, Any}()

        for name in keys(observation.field_time_serieses)
            loc = LX, LY, LZ = location(observation.field_time_serieses[name])
            topo = topology(grid)

            field_time_series = time_series_collector.field_time_serieses[name]

            raw_data = parent(field_time_series.data)
            data = OffsetArray(view(raw_data, :, j:j, :, :), 0, 0, -Hz, 0)

            time_series = FieldTimeSeries{LX, LY, LZ, InMemory}(data, CPU(), grid, nothing, times)
            time_serieses[name] = time_series
        end

        # Convert to NamedTuple
        time_serieses = NamedTuple(name => time_series for (name, time_series) in time_serieses)

        batch_output = OneDimensionalTimeSeries(time_serieses,
                                                grid,
                                                times,
                                                nothing,
                                                nothing,
                                                observation.normalization) 

        push!(transposed_output, batch_output)
    end

    return transposed_output
end

function drop_y_dimension(grid::RegularRectilinearGrid{<:Any, <:Flat, <:Flat, <:Bounded})
    new_size = ColumnEnsembleSize(Nz=grid.Nz, ensemble=(grid.Nx, 1), Hz=grid.Hz)
    z_domain = (grid.zF[1], grid.zF[grid.Nz])
    new_grid = RegularRectilinearGrid(size=new_size, z=z_domain, topology=(Flat, Flat, Bounded))
    return new_grid
end

end # module
