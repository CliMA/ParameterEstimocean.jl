function InverseProblem(data_batch::OneDimensionalTimeSeriesObservations, parameters::Parameters{UnionAll}; 
                                architecture = CPU(),
                            relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0),
                               ensemble_size = 1, 
                                          Δt = 60.0,
                                          Nz = 64)

    model = CATKEVerticalDiffusivityModel.EnsembleModel(td_batch; 
                                                        architecture = architecture,
                                                        N_ens = ensemble_size, 
                                                        parameter_specific_kwargs[parameters.RelevantParameters]...)

    simulation = Simulation(model; Δt = Δt, stop_time = 0.0)
    pop!(simulation.diagnostics, :nan_checker)

    # Set model to custom defaults
    set!(model, custom_defaults(model, parameters.RelevantParameters))

    default_parameters = custom_defaults(model, parameters.ParametersToOptimize)

    loss = LossFunction(simulation, data_batch; 
                        data_weights=[1.0 for data in data_batch],
                        relative_weights)

    return InverseProblem(data_batch, simulation, relative_weights, loss, default_parameters, parameters)
end

OneDimensionalTimeSeries(LESdata)

function InverseProblem(LESdata, parameters::Parameters{UnionAll}; kwargs...)

    data_batch = OneDimensionalTimeSeries.(values(LESdata); grid_type=ColumnEnsembleGrid, Nz=kwargs.Nz)

    return InverseProblem(data_batch, parameters; kwargs...)
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