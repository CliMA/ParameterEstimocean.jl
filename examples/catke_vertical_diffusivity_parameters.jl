using CUDA
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity,
                MixingLength, SurfaceTKEFlux, VerticallyImplicitTimeDiscretization

parameter_guide = Dict(

     :Cᴰ   => (
          name = "Dissipation parameter (TKE equation)",
          latex = L"C^D",
          default = 2.9079,
          bounds = (0.0, 10.0)),

     :Cᴸᵇ  => (
          name = "Mixing length parameter",
          latex = L"C^{\ell}_b",
          default = 1.1612,
          bounds = (0.0, 7.0)),
     
     :Cᵟu => (
          name = "Ratio of mixing length to grid spacing",
          latex = L"C^{\delta}u",
          default = 0.5,
          bounds = (0.0, 2.0)),

     :Cᵟc => (
          name = "Ratio of mixing length to grid spacing",
          latex = L"C^{\delta}c",
          default = 0.5,
          bounds = (0.0, 2.0)),

     :Cᵟe => (
          name = "Ratio of mixing length to grid spacing",
          latex = L"C^{\delta}e",
          default = 0.5,
          bounds = (0.0, 2.0)),

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

@free_parameters(CATKEParametersRiDependent,
                 Cᵟu, Cᵟc, Cᵟe,
                 CᴷRiʷ, CᴷRiᶜ,
                 Cᴷu⁻, Cᴷuʳ, Cᴷc⁻, Cᴷcʳ, Cᴷe⁻, Cᴷeʳ,
                 Cᴰ, Cᴸᵇ, Cᵂu★, CᵂwΔ)

@free_parameters(CATKEParametersRiIndependent,
                 Cᵟu, Cᵟc, Cᵟe,
                 Cᴷu⁻, Cᴷc⁻, Cᴷe⁻,
                 Cᴰ, Cᴸᵇ, Cᵂu★, CᵂwΔ)

@free_parameters(CATKEParametersRiDependentConvectiveAdjustment,
                 Cᵟu, Cᵟc, Cᵟe,
                 CᴷRiʷ, CᴷRiᶜ,
                 Cᴬu, Cᴬc, Cᴬe,
                 Cᴷu⁻, Cᴷuʳ, Cᴷc⁻, Cᴷcʳ, Cᴷe⁻, Cᴷeʳ,
                 Cᴰ, Cᴸᵇ, Cᵂu★, CᵂwΔ)

@free_parameters(CATKEParametersRiIndependentConvectiveAdjustment,
                 Cᵟu, Cᵟc, Cᵟe,
                 Cᴷu⁻, Cᴷc⁻, Cᴷe⁻,
                 Cᴬu, Cᴬc, Cᴬe,
                 Cᴰ, Cᴸᵇ, Cᵂu★, CᵂwΔ)

free_parameter_options = Dict(
    "CATKEParametersRiDependent" => CATKEParametersRiDependent,
#     "TKEFreeConvection" => TKEFreeConvection,
#     "TKEBCParameters" => TKEBCParameters,
)

parameter_specific_kwargs = Dict(
   CATKEParametersRiDependent => (mixing_length = MixingLength(Cᴬu=0.0, Cᴬc=0.0, Cᴬe=0.0),
                               ),

   CATKEParametersRiIndependent => (mixing_length = MixingLength(Cᴷuʳ=0.0, Cᴷcʳ=0.0, Cᴷeʳ=0.0,
                                                                Cᴬu=0.0, Cᴬc=0.0, Cᴬe=0.0),
                               ),

   CATKEParametersRiDependentConvectiveAdjustment => (mixing_length = MixingLength(),
                               ),

   CATKEParametersRiIndependentConvectiveAdjustment => (mixing_length = MixingLength(Cᴷuʳ=0.0, Cᴷcʳ=0.0, Cᴷeʳ=0.0),
                               ),
)