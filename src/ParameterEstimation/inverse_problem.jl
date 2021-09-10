using Oceananigans: AbstractModel

get_model_closure(model::AbstractModel) = get_model_closure(model.closure)
get_model_closure(closure) = closure
get_model_closure(closure::AbstractArray) = CUDA.@allowscalar closure[1, 1]

defaults(model::AbstractModel, RelevantParameters) = DefaultFreeParameters(get_model_closure(model), RelevantParameters)

function InverseProblem(observations::OneDimensionalTimeSeriesBatch, simulation::Simulation, parameters::Parameters{UnionAll}; transformation = )

    simulation = Simulation(model; Δt = Δt, stop_time = 0.0)
    pop!(simulation.diagnostics, :nan_checker)

    # Set model to custom defaults
    set!(model, custom_defaults(model, parameters.RelevantParameters))

    default_parameters = custom_defaults(model, parameters.ParametersToOptimize)

    return InverseProblem(observations, simulation, parameters)
end

function InverseProblem(LESdata, parameters::Parameters{UnionAll}; kwargs...)

    observations = OneDimensionalTimeSeries.(values(LESdata); grid_type=OneDimensionalEnsembleGrid, Nz=kwargs.Nz)

    return InverseProblem(observations, parameters; kwargs...)
end

function InverseProblem(data, ip::InverseProblem; 
                        architecture = ip.simulation.model.architecture,
                        relative_weights = ip.relative_weights,
                        ensemble_size = ip.simulation.model.grid.Nx,
                        Nz = ip.simulation.model.grid.Nz,
                        Δt = ip.simulation.Δt)

    return InverseProblem(data, ip.parameters;
                          architecture = architecture,
                          relative_weights = relative_weights,
                          ensemble_size = ensemble_size,
                          Nz = Nz,
                          Δt = Δt)
end