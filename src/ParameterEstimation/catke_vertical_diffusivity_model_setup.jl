function InverseProblem(LESdata, p::Parameters{UnionAll}; 
                                architecture = CPU(),
                            relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0),
                               ensemble_size = 1, 
                                          Nz = 64, 
                                          Δt = 60.0)

    td_batch = [TruthData(LEScase; grid_type=ColumnEnsembleGrid, Nz=Nz) for LEScase in values(LESdata)]

    model = CATKEVerticalDiffusivityModel.EnsembleModel(td_batch; 
                                                        architecture = architecture,
                                                        N_ens = ensemble_size, 
                                                        parameter_specific_kwargs[p.RelevantParameters]...)

    loss = LossFunction(model, td_batch, Δt, p.ParametersToOptimize; 
                        data_weights=[1.0 for td in td_batch],
                        relative_weights)

    # Set model to custom defaults
    set!(model, custom_defaults(model, p.RelevantParameters))

    default_parameters = custom_defaults(model, p.ParametersToOptimize)

    return InverseProblem(td_batch, model, relative_weights, loss, default_parameters)
end
