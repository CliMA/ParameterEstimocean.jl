
tke_fields(datum) = !(datum.stressed) ? (:b, :e) :
                    !(datum.rotating) ? (:b, :u, :e) :
                                        (:b, :u, :v, :e)

function get_loss(LEScase, td::Union{TruthData, BatchTruthData}, p::Parameters, relative_weights;
                                        # ParameterizedModel
                                                            Δt = 10.0,
                                        # TKE-specific kwargs:
                                            diffusivity_scaling = RiDependentDiffusivityScaling(),
                                        dissipation_parameter = 2.91,
                                        mixing_length_parameter = 1.16,
                                            time_discretization = VerticallyImplicitTimeDiscretization()
                                        )

    model = TKEMassFluxModel.ParameterizedModel(td, Δt;
                                        diffusivity_scaling = diffusivity_scaling,
                                      dissipation_parameter = dissipation_parameter,
                                    mixing_length_parameter = mixing_length_parameter,
                                              # surface_model = surface_model,
                                        time_discretization = time_discretization
                                         )
    
    set!(model, td, 1)

    fields = tke_fields(LEScase)

    relative_weights = [relative_weights[field] for field in fields]

    loss_function = init_loss_function(model, td, LEScase.first, LEScase.last,
                                        fields, relative_weights)

    
    loss = LossContainer(model, td, loss_function)

    # Set model to custom defaults
    set!(loss.model, custom_defaults(loss.model, p.RelevantParameters))

    default_parameters = custom_defaults(loss.model, p.ParametersToOptimize)
    return loss, default_parameters
end

function dataset(LESdata, p::Parameters{UnionAll}; 
                        relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0), 
                               grid_type = ZGrid, 
                                      Nz = 64,
                                      Δt = 60.0)

    if typeof(LESdata) <: NamedTuple

        # TruthData object containing LES data coarse-grained to a grid of size `N`
        # Coarse-graining the data at this step saves having to coarse-grain each time the loss is calculated
        td = TruthData(LEScase.filename; grid_type=grid_type, Nz=Nz)

        # Single simulation
        loss, default_parameters = get_loss(LESdata, td, p, relative_weights; Δt=Δt, 
                                                parameter_specific_kwargs[p.RelevantParameters]...)

    else

        td_batch = [TruthData(LEScase.filename; grid_type=grid_type, Nz=Nz) for LEScase in values(LESdata)]

        # Batched
        batch = []
        default_parameters = nothing

        for (i, LEScase) in enumerate(values(LESdata))
 
            loss, default_parameters = get_loss(LEScase, td_batch[i], p, relative_weights; Δt=Δt, 
                                                    parameter_specific_kwargs[p.RelevantParameters]...)
            push!(batch, loss)
        end
        loss = BatchedLossContainer([loss for loss in batch],
                                            weights=[1.0 for d in LESdata])
    end

    loss_wrapper(θ::Vector) = loss(p.ParametersToOptimize(θ))
    loss_wrapper(θ::FreeParameters) = loss(θ)

    return DataSet(LESdata, relative_weights, loss_wrapper, default_parameters)
end

function ensemble_dataset(LESdata, p::Parameters{UnionAll}; 
                                relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0), 
                                   ensemble_size = 1, 
                                              Nz = 64, 
                                              Δt = 60.0)

    td_batch = [TruthData(LEScase.filename; grid_type=ZGrid, Nz=Nz) for LEScase in values(LESdata)]

    first_targets = getproperty.(LESdata, :first_target)
    last_targets = getproperty.(LESdata, :last_target)

    model = ParameterizedModel(td_batch, Δt; N_ens = ensemble_size, kwargs...)

    # Build loss container of type `EnsembleLossContainer`
    loss = init_loss_function(model, td_batch, 
                        first_targets, last_targets, fields, relative_weights; weights=[1.0 for b in batch])

    loss_wrapper(θ::Vector) = loss(p.ParametersToOptimize(θ))
    loss_wrapper(θ::FreeParameters) = loss(θ)
end
