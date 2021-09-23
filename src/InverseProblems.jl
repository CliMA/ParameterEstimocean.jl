module InverseProblems

using ..Observations: obs_str, OneDimensionalTimeSeries, initialize_simulation!, FieldTimeSeriesCollector
using ..TurbulenceClosureParameters: free_parameters_str, update_closure_ensemble_member!

using Oceananigans: short_show, run!, fields
using Oceananigans.Fields: interior
using Oceananigans.Grids: Flat, Bounded, 
                          Face, Center,
                          RegularRectilinearGrid

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
    InverseProblem(observations, simluation, free_parameters; output_map=ConcatenatedOutputMap())

Return an `InverseProblem`.
"""
function InverseProblem(observations, simulation, free_parameters; output_map=ConcatenatedOutputMap(), time_series_collector=nothing)

    if isnothing(time_series_collector) # attempt to construct automagically
        simulation_fields = fields(simulation.model)
        collected_fields = NamedTuple(name => simulation_fields[name] for name in keys(observations.fields))
        time_series_collector = FieldTimeSeriesCollector(collected_fields, observations.times)
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
              "├── free_parameters: $(free_parameters_str(ip.free_parameters))",
              "└── output map: $out_map_str")

    return nothing
end

tupify_parameters(ip, θ) = NamedTuple{ip.free_parameters.names}(Tuple(θ))

#####
##### Forward map evaluation given vector-of-vector (one parameter vector for each ensemble member)
#####

const OneDimensionalEnsembleGrid = RegularRectilinearGrid{<:Any, Flat, Flat, Bounded}

# ensemble_size(grid::OneDimensionalEnsembleGrid) = grid.Nx
# batch_size(grid::OneDimensionalEnsembleGrid) = grid.Ny

n_ensemble(grid::OneDimensionalEnsembleGrid) = grid.Nx
n_observations(grid::OneDimensionalEnsembleGrid) = grid.Ny
n_z(grid::OneDimensionalEnsembleGrid) = grid.Nz

function forward_map(ip::InverseProblem, θ::Vector{<:NamedTuple})
    observations = ip.observations
    simulation = ip.simulation
    closures = simulation.model.closure

    for p in 1:length(θ)
        update_closure_ensemble_member!(closures, p, θ[p])
    end

    initialize_simulation!(simulation, observations, ip.time_series_collector)
    run!(simulation)

    return map_output(ip.output_map, ip.time_series_collector, observations)
end

#=
function forward_map(ip, θ::NamedTuple)

    single_member_grid = 
    single_member_simulation = 
    
    single_member_ip = InverseProblem(ip.observations,
                                      single_member_simulation,
                                      ip.free_parameters;
                                      output_map = ip.output_map,
                                      time_series_collector = ip.time_series_collector)

    return forward_map(single_member_ip, [θ])
end
=#

forward_map(ip, θ::Matrix) = forward_map(ip, [tupify_parameters(ip, θ[:, i]) for i in 1:size(θ, 2)])

(ip::InverseProblem)(θ) = forward_map(ip, θ)

function normalize!(field_data, observations::Vector{<:OneDimensionalTimeSeries}, field_name)

    for (batch_index, observation) in enumerate(observations)
        normalize!(view(field_data, :, batch_index, :), observation, field_name)
    end

    return nothing
end

normalize!(field_data, observation::OneDimensionalTimeSeries, field_name) =
    normalize!(field_data, observation.normalization[field_name])

observation_field_names(observation::OneDimensionalTimeSeries) = keys(observation.fields)

observation_field_names(observations::Vector{<:OneDimensionalTimeSeries}) =
    observation_field_names(observations[1])

# Returns an ``$(n_t n_z m, n_\text{ensemble})$`` array, where 
# ``$m=\sum_{i=1}^{n_\text{observations}}N_\text{fields, i}$``.
function map_output(::ConcatenatedOutputMap, time_series_collector, observations)

    grid = time_series_collector.grid
    stacked = reshape([], n_ensemble(grid), 0)

    for (time_index, time) in enumerate(time_series_collector.times)
        for field_name in observation_field_names(observations)

            field_time_series = time_series_collector.field_time_serieses[field_name]
            field = field_time_series[time_index]
            interior_field_data = Array(interior(field))
            normalize!(interior_field_data, observations, field_name)

            Nx, Ny, Nz = size(interior_field_data)
            flattened_field_data = reshape(interior_field_data, Nx, Ny * Nz)

            stacked = hcat(stacked, flattened_field_data)
        end
    end

    return stacked'
end

# Either
map_observations(::ConcatenatedOutputMap, observations::Vector{<:OneDimensionalTimeSeries}) =
    map_output(ConcatenatedOutputMap(), reshape(observations, 1, length(observations)), nothing)

# or new code

end # module