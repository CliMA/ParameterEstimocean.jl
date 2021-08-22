

function get_loss(td::TruthData, p::Parameters, relative_weights;
                                        # ParameterizedModel
                                                            Δt = 10.0,
                                        # TKE-specific kwargs:
                                        #                     Cᴰ = 2.91,
                                        #          mixing_length = MixingLength(),
                                        #       surface_tke_flux = SurfaceTKEFlux(),
                                        #    time_discretization = VerticallyImplicitTimeDiscretization(),
                                                       kwargs...
                                        )

    model = TKEMassFluxModel.ParameterizedModel(td, Δt; kwargs...)
    
    set!(model, td, 1)

    relative_weights = [relative_weights[field] for field in td.relevant_fields]
    loss_function = init_loss_function(model, td, relative_weights)
    
    loss = LossContainer(model, td, loss_function)

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

    if typeof(LESdata) <: NamedTuple

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
        model = [b.model for b in loss.batch]

    end

    loss_wrapper(θ::Vector) = loss(p.ParametersToOptimize(θ))
    loss_wrapper(θ::FreeParameters) = loss(θ)

    return DataSet(LESdata, data, model, relative_weights, loss_wrapper, default_parameters)
end

function ensemble_dataset(LESdata, p::Parameters{UnionAll}; 
                                relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0), 
                                   ensemble_size = 1, 
                                              Nz = 64, 
                                              Δt = 60.0)

    td_batch = [TruthData(LEScase.filename; grid_type=ColumnEnsembleGrid, Nz=Nz) for LEScase in values(LESdata)]

    model = ParameterizedModel(td_batch, Δt; N_ens=ensemble_size, 
                                            parameter_specific_kwargs[p.RelevantParameters]...)

    loss = EnsembleLossContainer(model, td_batch; data_weights=[1.0 for td in td_batch],
                                                           relative_weights)

    # Set model to custom defaults
    set!(loss.model, custom_defaults(loss.model, p.RelevantParameters))

    default_parameters = custom_defaults(loss.model, p.ParametersToOptimize)

    loss_wrapper(θ::Vector{<:Number}) = loss(p.ParametersToOptimize(θ))
    loss_wrapper(θ::FreeParameters) = loss(θ)

    loss_wrapper(θ::Vector{<:Vector}) = loss(p.ParametersToOptimize.(θ))
    loss_wrapper(θ::Vector{<:FreeParameters}) = loss(θ)

    return DataSet(LESdata, loss.data_batch, loss.model, relative_weights, loss_wrapper, default_parameters)
end
