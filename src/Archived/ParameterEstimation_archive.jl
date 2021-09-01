

function get_loss(td::TruthData, p::Parameters, relative_weights; 
                                    Δt = 10.0, kwargs...)

    model = CATKEVerticalDiffusivityModel.HydrostaticFreeSurfaceModel(td; kwargs...)
    
    set!(model, td, 1)

    relative_weights = [relative_weights[field] for field in td.relevant_fields]
    loss_function = init_loss_function(model, td, relative_weights)
    
    loss = LossContainer(model, td, loss_function, Δt)

    # Set model to custom defaults
    set!(loss.model, custom_defaults(loss.model, p.RelevantParameters))

    default_parameters = custom_defaults(loss.model, p.ParametersToOptimize)
    return loss, default_parameters
end

function dataset(LESdata, p::Parameters{UnionAll}; 
                        relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0), 
                               grid_type = ColumnEnsembleGrid, 
                                      Nz = 64,
                                      Δt = 60.0)

    if LESdata isa NamedTuple

        # TruthData object containing LES data coarse-grained to a grid of size `N`
        # Coarse-graining the data at this step saves having to coarse-grain each time the loss is calculated
        td = TruthData(LESdata; grid_type=grid_type, Nz=Nz)

        # Single simulation
        loss, default_parameters = get_loss(td, p, relative_weights; Δt=Δt,
                                                parameter_specific_kwargs[p.RelevantParameters]...)

        data = [td]
        model = [loss.model]

    else

        data = [TruthData(LEScase; grid_type=grid_type, Nz=Nz) for LEScase in values(LESdata)]

        # Batched
        batch = []
        default_parameters = nothing

        for (i, LEScase) in enumerate(values(LESdata))
 
            loss, default_parameters = get_loss(data[i], p, relative_weights; Δt=Δt,
                                                    parameter_specific_kwargs[p.RelevantParameters]...)
            push!(batch, loss)

        end

        loss = BatchedLossContainer([loss for loss in batch],
                                            weights=[1.0 for d in LESdata])
        model = loss.batch[1].model

    end

    loss_wrapper(θ::Vector) = loss(p.ParametersToOptimize(θ))
    loss_wrapper(θ::FreeParameters) = loss(θ)

    return DataSet(LESdata, data, model, relative_weights, loss_wrapper, default_parameters)
end