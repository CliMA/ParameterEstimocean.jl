module InverseProblems

using ..Observations: obs_str
using ..TurbulenceClosureParameters: free_parameters_str, update_closure_ensemble_member!

using Oceananigans: short_show

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

struct InverseProblem{F, O, S, P}
    observations :: O
    simulation :: S
    free_parameters :: P
    output_map :: F
end

"""
    InverseProblem(observations, simluation, free_parameters; output_map=ConcatenatedOutputMap())

Return an `InverseProblem`.
"""
function InverseProblem(observations, simulation, free_parameters; output_map=ConcatenatedOutputMap())
    return InverseProblem(observations, simulation, free_parameters, output_map)
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

#####
##### Forward map evaluation given vector-of-vector (one parameter vector for each ensemble member)
#####

function (ip::InverseProblem{<:ConcatenatedOutputMap})(θ)
    observations = ip.observations
    simulation = ip.simulation
    free_parameters = ip.free_parameters

    closures = simulation.model.closure

    for p in 1:length(θ)
        # Extract parameters for ensemble member p and convert to NamedTuple
        θp = NamedTuple{free_parameters.names}(Tuple(θ[p]))
        update_closure_ensemble_member!(closures, p, θp)
    end

    initialize_simulation!(simulation, observations)

    run!(simulation)

    return nothing
end

end # module