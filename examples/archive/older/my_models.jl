
@free_parameters(TKEConvectiveAdjustmentIndependentDiffusivites,
                  Cᴷc, Cᴷe, Cᴰ, Cᴸᵇ, CʷwΔ, Cᴬ)
@free_parameters(TKEConvectiveAdjustmentIndependentDiffusivites_ParametersToOptimize,
                  Cᴷc, Cᴷe, Cᴰ, Cᴸᵇ, CʷwΔ, Cᴬ)

@free_parameters TKEFreeConvectionIndependentDiffusivites Cᴷc Cᴷe Cᴰ Cᴸᵇ CʷwΔ
@free_parameters TKEFreeConvectionIndependentDiffusivites_ParametersToOptimize Cᴷc Cᴷe Cᴰ Cᴸᵇ CʷwΔ

@free_parameters(TKEFreeConvection,
                 CᴷRiʷ, CᴷRiᶜ,
                 Cᴷc⁻, Cᴷc⁺,
                 Cᴷe⁻, Cᴷe⁺,
                 Cᴰ, Cᴸᵇ, CʷwΔ)

@free_parameters(TKEConvectiveAdjustment,
                 CᴷRiʷ, CᴷRiᶜ,
                 Cᴷc⁻, Cᴷc⁺,
                 Cᴷe⁻, Cᴷe⁺,
                 Cᴰ, Cᴸᵇ, CʷwΔ, Cᴬ)

function build_loss(model, tdata)
        # Loss Function
        Nt = length(tdata.t)
        Δt = tdata.t[2] - tdata.t[1]
        targets = Int(Δt*12/60)+1:Nt # cut out the first 2 hours
        # targets = 1:Nt
        loss_function = LossFunction(model, tdata,
                                    fields=(:T,),
                                    targets=targets,
                                    weights=[1.0,],
                                    time_series = TimeSeriesAnalysis(tdata.t[targets], TimeAverage()),
                                    profile = ValueProfileAnalysis(model.grid)
                                    )
        loss = LossContainer(model, tdata, loss_function)
        
        return [loss_function, loss]
end

function custom_defaults(model, RelevantParameters)
    fields = fieldnames(RelevantParameters)
    defaults = DefaultFreeParameters(model, RelevantParameters)
    function set!(defaults, x, d)
         if x in fields; setfield!(defaults, x,d); end
    end

    set!(defaults, :Cᴰ,  0.01)
    set!(defaults, :Cᴸᵇ, 0.02)
    set!(defaults, :CʷwΔ, 5.0)

    # Independent diffusivities
    set!(defaults, :Cᴷc, 0.5)
    set!(defaults, :Cᴷe, 0.02)

    return defaults
end

function get_bounds(model, ParametersToOptimize)
    fields = fieldnames(ParametersToOptimize)
    bounds = ParametersToOptimize(((0.01, 2.0) for p in default_parameters)...)

    function set!(bounds, x, b)
        if x in fields; setfield!(bounds, x, b); end
    end

    set!(bounds, :Cᴰ,  (0.01, 5.0))
    set!(bounds, :Cᴸᵇ, (0.01, 0.5))
    set!(bounds, :CʷwΔ, (0.01, 10.0))

    # Convective adjustment
    set!(bounds, :Cᴬ,  (0.01, 40.0))

    # Independent diffusivities
    set!(bounds, :Cᴷc, (0.005, 2.0))
    set!(bounds, :Cᴷe, (0.005, 0.5))

    # RiDependentDiffusivities
    set!(bounds, :CᴷRiʷ,  (0.01, 40.0))
    set!(bounds, :CᴷRiᶜ,  (0.005, 0.5))
    set!(bounds, :Cᴷc⁻,  (0.005, 0.5))
    set!(bounds, :Cᴷc⁺,  (0.005, 0.5))
    set!(bounds, :Cᴷc⁺,  (0.005, 0.5))
    set!(bounds, :Cᴷe⁻,  (0.005, 0.5))
    set!(bounds, :Cᴷe⁺,  (0.005, 0.5))
    set!(bounds, :Cᴷe⁺,  (0.005, 0.5))

    return bounds
end


struct MyParameterizedModel{td, MM, PO, DP, BB}
    tdata :: td
    model :: MM
    ParametersToOptimize ::PO
    default_parameters :: DP
    bounds :: BB
end

function conv_adj_independent_diffusivites(file)

    # ParameterizedModel and data
    tdata = OneDimensionalTimeSeries(file)
    model = ParameterizedModel(tdata, 1minute, N=32,
                        convective_adjustment = TKEMassFlux.FluxProportionalConvectiveAdjustment(),
                        eddy_diffusivities = TKEMassFlux.IndependentDiffusivities()
                        )

    set!(model, custom_defaults(model, ConvectiveAdjustmentIndependentDiffusivitesTKEParameters))
    ParametersToOptimize = ConvectiveAdjustmentIndependentDiffusivitesTKEParameters_ParametersToOptimize
    default_parameters = custom_defaults(model, ParametersToOptimize)
    bounds = get_bounds(model, ParametersToOptimize)

    return MyParameterizedModel(tdata, model, ParametersToOptimize, default_parameters, bounds)
end


function tke_free_convection_independent_diffusivities(file)
    # no convective adjustment
    # simple mixing length

    # ParameterizedModel and data
    tdata = OneDimensionalTimeSeries(file)
    model = ParameterizedModel(tdata, 1minute, N=32,
                        convective_adjustment = nothing,
                        eddy_diffusivities = TKEMassFlux.IndependentDiffusivities()
                        )

    set!(model, custom_defaults(model, TKEFreeConvectionIndependentDiffusivites))
    ParametersToOptimize = TKEFreeConvectionIndependentDiffusivites_ParametersToOptimize
    default_parameters = custom_defaults(model, ParametersToOptimize)
    bounds = get_bounds(model, ParametersToOptimize)

    return MyParameterizedModel(tdata, model, ParametersToOptimize, default_parameters, bounds)
end

function tke_free_convection(file)
    # no convective adjustment
    # simple mixing length

    # ParameterizedModel and data
    tdata = OneDimensionalTimeSeries(file)
    model = ParameterizedModel(tdata, 1minute, N=32,
                        convective_adjustment = nothing,
                        eddy_diffusivities = TKEMassFlux.RiDependentDiffusivities()
                        )

    set!(model, custom_defaults(model, TKEFreeConvection))
    ParametersToOptimize = TKEFreeConvection_ParametersToOptimize
    default_parameters = custom_defaults(model, ParametersToOptimize)
    bounds = get_bounds(model, ParametersToOptimize)

    return MyParameterizedModel(tdata, model, ParametersToOptimize, default_parameters, bounds)
end

function conv_adj_ri_dependent_diffusivites(file)

    # ParameterizedModel and data
    tdata = OneDimensionalTimeSeries(file)
    model = ParameterizedModel(tdata, 1minute, N=32,
                        convective_adjustment = TKEMassFlux.FluxProportionalConvectiveAdjustment(),
                        eddy_diffusivities = TKEMassFlux.RiDependentDiffusivities()
                        )

    set!(model, custom_defaults(model, TKEConvectiveAdjustment))
    ParametersToOptimize = TKEConvectiveAdjustment_ParametersToOptimize
    default_parameters = custom_defaults(model, ParametersToOptimize)
    bounds = get_bounds(model, ParametersToOptimize)

    return MyParameterizedModel(tdata, model, ParametersToOptimize, default_parameters, bounds)
end
