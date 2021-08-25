parameter_guide = Dict(

     :Cᴰ   => (
          name = "Dissipation parameter (TKE equation)",
          latex = L"C^D",
          default = 2.9079,
          bounds = (0.0, 10.0)),

     :Cᴷu  => (
          name = "Velocity diffusivity (defunct)",
          latex = L"C^K_U",
          default = 0.1243,
          bounds = (0.0, 3.0)),

     :Cᴷc  => (
          name = "Tracer diffusivity (defunct)",
          latex = L"C^K_C",
          default = 0.09518,
          bounds = (0.0, 3.0)),

     :Cᴷe  => (
          name = "TKE diffusivity (defunct)",
          latex = L"C^K_E",
          default = 0.055,
          bounds = (0.0, 3.0)),

     :Cᴸᵇ  => (
          name = "Mixing length parameter",
          latex = L"C^{\ell}_b",
          default = 1.1612,
          bounds = (0.0, 7.0)),
     
     :Cᵟu => (
          name = "Ratio of mixing length to grid spacing",
          latex = L"C^{\delta}u",
          default = 0.5,
          bounds = (0.0, 2.0),
     ),

     :Cᵟc => (
          name = "Ratio of mixing length to grid spacing",
          latex = L"C^{\delta}c",
          default = 0.5,
          bounds = (0.0, 2.0),
     ),

     :Cᵟe => (
          name = "Ratio of mixing length to grid spacing",
          latex = L"C^{\delta}e",
          default = 0.5,
          bounds = (0.0, 2.0),
     ),

     :Cᵂu★  => (
          name = "Mixing length parameter",
          latex = L"C^W_{u\star}",
          default = 3.6188,
          bounds = (0.0, 20.0)),

     :CʷwΔ  => (
          name = "Mixing length parameter",
          latex = L"C^ww\Delta",
          default = 1.3052,
          bounds = (0.0, 5.0)),

     :CᴷRiʷ  => (
          name = "Stability function parameter",
          latex = L"C^KRi^w",
          default = 0.7213,
          bounds = (0.0, 6.0)),

     :CᴷRiᶜ  => (
          name = "Stability function parameter",
          latex = L"C^KRi^c",
          default = 0.7588,
          bounds = (-1.5, 4.0)),

     :Cᴷu⁻  => (
          name = "Velocity diffusivity LB",
          latex = L"C^Ku^-",
          default = 0.1513,
          bounds = (0.0, 1.0)),

     :Cᴷu⁺  => (
          name = "Velocity diffusivity UB (defunct)",
          latex = L"C^Ku^+",
          default = 0.7337,
          bounds = (0.0, 5.0)),

     :Cᴷuʳ  => (
          name = "Velocity diffusivity (UB-LB)/LB",
          latex = L"C^Ku^r",
          default = 3.8493,
          bounds = (0.0, 100.0)),

     :Cᴷc⁻  => (
          name = "Velocity diffusivity LB",
          latex = L"C^Kc^-",
          default = 0.3977,
          bounds = (0.0, 3.0)),

     :Cᴷc⁺  => (
          name = "Velocity diffusivity UB (defunct)",
          latex = L"C^Kc^+",
          default = 1.7738,
          bounds = (0.0, 3.0)),

     :Cᴷcʳ  => (
          name = "Velocity diffusivity (UB-LB)/LB",
          latex = L"C^Kc^r",
          default = 3.4601,
          bounds = (0.0, 100.0)),

     :Cᴷe⁻  => (
          name = "Velocity diffusivity LB",
          latex = L"C^Ke^-",
          default = 0.1334,
          bounds = (0.0, 3.0)),

     :Cᴷe⁺  => (
          name = "Velocity diffusivity UB (defunct)",
          latex = L"C^Ke^+",
          default = 1.2247,
          bounds = (0.0, 3.0)),

     :Cᴷeʳ  => (
          name = "Velocity diffusivity (UB-LB)/LB",
          latex = L"C^Ke^r",
          default = 8.1806,
          bounds = (0.0, 100.0)),

     :Cᴬu  => (
          name = "Convective adjustment velocity parameter",
          latex = L"C^A_U",
          default = 0.0057,
          bounds = (0.0, 0.3)),

     :Cᴬc  => (
          name = "Convective adjustment tracer parameter",
          latex = L"C^A_C",
          default = 0.6706,
          bounds = (0.0, 2.0)),

     :Cᴬe => (
          name = "Convective adjustment TKE parameter",
          latex = L"C^A_E",
          default = 0.2717,
          bounds = (0.0, 2.0)),
)

# For scenarios involving stresses
@free_parameters(TKEParametersRiDependent,
                 Cᵟu, Cᵟc, Cᵟe,
                 CᴷRiʷ, CᴷRiᶜ,
                 Cᴷu⁻, Cᴷuʳ, Cᴷc⁻, Cᴷcʳ, Cᴷe⁻, Cᴷeʳ,
                 Cᴰ, Cᴸᵇ, Cᵂu★, CᵂwΔ)

@free_parameters(TKEParametersRiIndependent,
                 Cᵟu, Cᵟc, Cᵟe,
                 Cᴷu⁻, Cᴷc⁻, Cᴷe⁻,
                 Cᴰ, Cᴸᵇ, Cᵂu★, CᵂwΔ)

@free_parameters(TKEParametersRiDependentConvectiveAdjustment,
                 Cᵟu, Cᵟc, Cᵟe,
                 CᴷRiʷ, CᴷRiᶜ,
                 Cᴬu, Cᴬc, Cᴬe,
                 Cᴷu⁻, Cᴷuʳ, Cᴷc⁻, Cᴷcʳ, Cᴷe⁻, Cᴷeʳ,
                 Cᴰ, Cᴸᵇ, Cᵂu★, CᵂwΔ)

@free_parameters(TKEParametersRiIndependentConvectiveAdjustment,
                 Cᵟu, Cᵟc, Cᵟe,
                 Cᴷu⁻, Cᴷc⁻, Cᴷe⁻,
                 Cᴬu, Cᴬc, Cᴬe,
                 Cᴰ, Cᴸᵇ, Cᵂu★, CᵂwΔ)

# For purely convective scenarios
# @free_parameters(TKEFreeConvection,
#                  Cᵟu, Cᵟc, Cᵟe,
#                  CᴷRiʷ, CᴷRiᶜ,
#                  Cᴷc⁻, Cᴷcʳ,
#                  Cᴷe⁻, Cᴷeʳ,
#                  Cᴰ, Cᴸᵇ, CᵂwΔ)

# @free_parameters(TKEBCParameters, Cᵂu★, CᵂwΔ)

free_parameter_options = Dict(
    "TKEParametersRiDependent" => TKEParametersRiDependent,
#     "TKEFreeConvection" => TKEFreeConvection,
#     "TKEBCParameters" => TKEBCParameters,
)

parameter_specific_kwargs = Dict(
   TKEParametersRiDependent => (mixing_length = MixingLength(Cᴬu=0.0, Cᴬc=0.0, Cᴬe=0.0),
                               ),

   TKEParametersRiIndependent => (mixing_length = MixingLength(Cᴷuʳ=0.0, Cᴷcʳ=0.0, Cᴷeʳ=0.0,
                                                                Cᴬu=0.0, Cᴬc=0.0, Cᴬe=0.0),
                               ),

   TKEParametersRiDependentConvectiveAdjustment => (mixing_length = MixingLength(),
                               ),

   TKEParametersRiIndependentConvectiveAdjustment => (mixing_length = MixingLength(Cᴷuʳ=0.0, Cᴷcʳ=0.0, Cᴷeʳ=0.0),
                               ),

#    TKEFreeConvection => (diffusivity_scaling = DD,
#                                ),

#    TKEBCParameters => (diffusivity_scaling = DD,
#                                ),
)

override_defaults = Dict(
#     TKEParametersRiDependent => [0.6487, 1.9231, 0.2739, 5.7999, 0.2573, 4.8146, 0.2941, 3.7099, 3.0376, 1.6998, 3.5992, 1.8507],
)

set_if_present!(obj, name, field) = name ∈ propertynames(obj) && setproperty!(obj, name, field)

function custom_defaults(model::AbstractModel, RelevantParameters)
    fields = fieldnames(RelevantParameters)

    mc = model.closure
    closure = typeof(mc) <: Matrix ? mc[1,1] : mc
    defaults = DefaultFreeParameters(closure, RelevantParameters)

    RelevantParameters ∈ keys(override_defaults) && return RelevantParameters(override_defaults[RelevantParameters])

    for (pname, info) in parameter_guide
          set_if_present!(defaults, pname, info.default)
    end

    return defaults
end

function get_bounds_and_variance(default_parameters; stds_within_bounds = 5)

    SomeFreeParameters = typeof(default_parameters).name.wrapper

    # Set bounds on free parameters
    bounds = SomeFreeParameters([(0.0, 10.0) for p in default_parameters]...)

    for (pname, info) in parameter_guide
        set_if_present!(bounds, pname, info.bounds)
    end

    # if stds_within_bounds = 3, 3 standard deviations to either side of the mean fits between the bounds
    variances = SomeFreeParameters((((bound[2] - bound[1])/(2 * stds_within_bounds))^2 for bound in bounds)...)

    variances = Array(variances)

    return bounds, variances
end