"Initialize a calibration run for the TKEBasedVerticalDiffusivity parameterization."
function init_tke_calibration(datapath;

                                # model grid
                                       grid_type = ZGrid, 
                                       grid_size = 64,

                                # ParameterizedModel
                                              Δt = 10.0,

                                # Loss function
                                    first_target = 5,
                                     last_target = nothing,
                                          fields = (:b, :u, :v, :e),
                                relative_weights = [1.0 for f in fields],

                                # TKE-specific kwargs:
                             diffusivity_scaling = RiDependentDiffusivityScaling(),
                           dissipation_parameter = 2.91,
                         mixing_length_parameter = 1.16,
                                   # surface_model = TKESurfaceFlux(),
                             time_discretization = VerticallyImplicitTimeDiscretization()

                              )

    # TruthData object containing LES data coarse-grained to a grid of size `N`
    # Coarse-graining the data at this step saves having to 
    td = TruthData(datapath; grid_type=grid_type, 
                             grid_size=grid_size)

    model = TKEMassFluxModel.ParameterizedModel(td, Δt;
                                        diffusivity_scaling = diffusivity_scaling,
                                      dissipation_parameter = dissipation_parameter,
                                    mixing_length_parameter = mixing_length_parameter,
                                              # surface_model = surface_model,
                                        time_discretization = time_discretization
                                         )

    set!(model, td, 1)

    return init_negative_log_likelihood(model, td, first_target, last_target,
                                        fields, relative_weights)
end

tke_fields(datum) = !(datum.stressed) ? (:b, :e) :
                    !(datum.rotating) ? (:b, :u, :e) :
                                        (:b, :u, :v, :e)

function get_nll(LEScase, p::Parameters, relative_weights; grid_type=ZGrid, grid_size=64, Δt=10.0)

    fields = tke_fields(LEScase)

    relative_weights_ = [relative_weights[field] for field in fields]
    nll = init_tke_calibration(LEScase.filename;
                                     grid_type = grid_type,
                                     grid_size = grid_size,
                                            Δt = Δt,
                                  first_target = LEScase.first,
                                   last_target = LEScase.last,
                                        fields = fields,
                              relative_weights = relative_weights_,
                              parameter_specific_kwargs[p.RelevantParameters]...
                            )

    # Set model to custom defaults
    set!(nll.model, custom_defaults(nll.model, p.RelevantParameters))

    default_parameters = custom_defaults(nll.model, p.ParametersToOptimize)
    return nll, default_parameters
end

function dataset(LESdata, p::Parameters{UnionAll}; relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0), grid_type=ZGrid, grid_size=64, Δt=60.0)

    if typeof(LESdata) <: NamedTuple

        # Single simulation
        nll, default_parameters = get_nll(LESdata, p, relative_weights; grid_type=grid_type, grid_size=grid_size, Δt=Δt)

    else

        # Batched
        batch = []
        default_parameters = nothing
        for LEScase in values(LESdata)
            nll, default_parameters = get_nll(LEScase, p, relative_weights; grid_type=grid_type, grid_size=grid_size, Δt=Δt)
            push!(batch, nll)
        end
        nll = BatchedNegativeLogLikelihood([nll for nll in batch],
                                            weights=[1.0 for d in LESdata])
    end

    # Loss wrapper for vector input
    nll_wrapper(θ::Vector) = nll(p.ParametersToOptimize(θ))

    return DataSet(LESdata, relative_weights, nll, nll_wrapper, default_parameters)
end
