pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using OceanTurbulenceParameterEstimation, LinearAlgebra, CairoMakie
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity

include("lesbrary_paths.jl")
include("visualize_profile_predictions.jl")
examples_path = joinpath(pathof(OceanTurbulenceParameterEstimation), "../../examples")
include(joinpath(examples_path, "intro_to_inverse_problems.jl"))

using LaTeXStrings

import OceanTurbulenceParameterEstimation.TurbulenceClosureParameters: closure_with_parameters

parameter_guide = Dict(:Cᴰ => (name = "Dissipation parameter (TKE equation)", latex = L"C^D",
        default = 2.9079, bounds = (0.0, 10.0)), :Cᴸᵇ => (name = "Mixing length parameter", latex = L"C^{\ell}_b",
        default = 1.1612, bounds = (0.0, 10.0)), :Cᵟu => (name = "Ratio of mixing length to grid spacing", latex = L"C^{\delta}u",
        default = 0.5, bounds = (0.0, 3.0)), :Cᵟc => (name = "Ratio of mixing length to grid spacing", latex = L"C^{\delta}c",
        default = 0.5, bounds = (0.0, 3.0)), :Cᵟe => (name = "Ratio of mixing length to grid spacing", latex = L"C^{\delta}e",
        default = 0.5, bounds = (0.0, 3.0)), :Cᵂu★ => (name = "Mixing length parameter", latex = L"C^W_{u\star}",
        default = 3.6188, bounds = (0.0, 20.0)), :CᵂwΔ => (name = "Mixing length parameter", latex = L"C^Ww\Delta",
        default = 1.3052, bounds = (0.0, 5.0)), :CᴷRiʷ => (name = "Stability function parameter", latex = L"C^KRi^w",
        default = 0.7213, bounds = (0.0, 5.0)), :CᴷRiᶜ => (name = "Stability function parameter", latex = L"C^KRi^c",
        default = 0.7588, bounds = (-1.5, 4.0)), :Cᴷu⁻ => (name = "Velocity diffusivity LB", latex = L"C^Ku^-",
        default = 0.1513, bounds = (0.0, 2.0)), :Cᴷuʳ => (name = "Velocity diffusivity (UB-LB)/LB", latex = L"C^Ku^r",
        default = 3.8493, bounds = (0.0, 50.0)), :Cᴷc⁻ => (name = "Velocity diffusivity LB", latex = L"C^Kc^-",
        default = 0.3977, bounds = (0.0, 5.0)), :Cᴷcʳ => (name = "Velocity diffusivity (UB-LB)/LB", latex = L"C^Kc^r",
        default = 3.4601, bounds = (0.0, 50.0)), :Cᴷe⁻ => (name = "Velocity diffusivity LB", latex = L"C^Ke^-",
        default = 0.1334, bounds = (0.0, 3.0)), :Cᴷeʳ => (name = "Velocity diffusivity (UB-LB)/LB", latex = L"C^Ke^r",
        default = 8.1806, bounds = (0.0, 50.0)), :Cᴬu => (name = "Convective adjustment velocity parameter", latex = L"C^A_U",
        default = 0.0057, bounds = (0.0, 0.2)), :Cᴬc => (name = "Convective adjustment tracer parameter", latex = L"C^A_C",
        default = 0.6706, bounds = (0.0, 2.0)), :Cᴬe => (name = "Convective adjustment TKE parameter", latex = L"C^A_E",
        default = 0.2717, bounds = (0.0, 2.0)),
)

bounds(name) = parameter_guide[name].bounds
default(name) = parameter_guide[name].default

function named_tuple_map(names, f)
    names = Tuple(names)
    return NamedTuple{names}(f.(names))
end

"""
    ParameterSet()

Parameter set containing the names `names` of parameters, and a 
NamedTuple `settings` mapping names of "background" parameters 
to their fixed values to be maintained throughout the calibration.
"""
struct ParameterSet{N,S}
    names::N
    settings::S
end

"""
    ParameterSet(names::Set; nullify = Set())

Construct a `ParameterSet` containing all of the information necessary 
to build a closure with the specified default parameters and settings,
given a Set `names` of the parameter names to be tuned, and a Set `nullify`
of parameters to be set to zero.
"""
function ParameterSet(names::Set; nullify = Set())
    zero_set = named_tuple_map(nullify, name -> 0.0)
    bkgd_set = named_tuple_map(keys(parameter_guide), name -> default(name))
    settings = merge(bkgd_set, zero_set) # order matters: `zero_set` overrides `bkgd_set`
    return ParameterSet(Tuple(names), settings)
end

names(ps::ParameterSet) = ps.names

###
### Define some convenient parameter sets based on the present CATKE formulation
###

required_params = Set([:Cᵟu, :Cᵟc, :Cᵟe, :Cᴰ, :Cᴸᵇ, :Cᵂu★, :CᵂwΔ, :Cᴷu⁻, :Cᴷc⁻, :Cᴷe⁻])
ri_depen_params = Set([:CᴷRiʷ, :CᴷRiᶜ, :Cᴷuʳ, :Cᴷcʳ, :Cᴷeʳ])
conv_adj_params = Set([:convective_κz, :background_κz, :Cᴬe])

convective_κz = 1,
background_κz = 1e-5,
convective_νz = 1e-3,
background_νz = 1e-4, parameter_set = ParameterSet(union(required_params, ri_depen_params))

# Pick a parameter set defined in `./parameters.jl`
parameter_set = CATKEParametersRiDependent
closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

# Pick the secret `true_parameters`
true_parameters = (Cᵟu = 0.5, CᴷRiʷ = 1.0, Cᵂu★ = 2.0, CᵂwΔ = 1.0, Cᴷeʳ = 5.0, Cᵟc = 0.5, Cᴰ = 2.0, Cᴷc⁻ = 0.5, Cᴷe⁻ = 0.2, Cᴷcʳ = 3.0, Cᴸᵇ = 1.0, CᴷRiᶜ = 1.0, Cᴷuʳ = 4.0, Cᴷu⁻ = 1.2, Cᵟe = 0.5)
true_closure = closure_with_parameters(closure, true_parameters)

# Generate and load synthetic observations
kwargs = (tracers = (:b, :e), stop_time = 60.0, Δt = 10.0)

observ_path1 = generate_synthetic_observations("perfect_model_observation1"; Qᵘ = 2e-5, Qᵇ = 2e-8, f₀ = 1e-4, closure = true_closure, kwargs...)
observ_path2 = generate_synthetic_observations("perfect_model_observation2"; Qᵘ = -5e-5, Qᵇ = 0, f₀ = -1e-4, closure = true_closure, kwargs...)
observation1 = SyntheticObservations(observ_path1, field_names = (:b, :e, :u, :v), normalize = ZScore)
observation2 = SyntheticObservations(observ_path2, field_names = (:b, :e, :u, :v), normalize = ZScore)
observations = [observation1, observation2]

# Set up ensemble model
ensemble_simulation, closure = build_ensemble_simulation(observations; Nensemble = 50)

# Build free parameters
build_prior(name) = ConstrainedNormal(0.0, 1.0, bounds(name) .* 0.5...)
free_parameters = FreeParameters(named_tuple_map(names(parameter_set), build_prior))

# Pack everything into Inverse Problem `calibration`
calibration = InverseProblem(observations, ensemble_simulation, free_parameters);

# Make sure the forward map evaluated on the true parameters matches the observation map
x = forward_map(calibration, true_parameters)[:, 1:1];
y = observation_map(calibration);
@show x == y

# visualize!(calibration, true_parameters;
#     field_names = [:u, :v, :b, :e],
#     directory = @__DIR__,
#     filename = "perfect_model_visual.png"
# )

# # Calibrate
# eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)
# params = iterate!(eki; iterations = 5)

# visualize!(calibration, params;
#     field_names = [:u, :v, :b, :e],
#     directory = @__DIR__,
#     filename = "perfect_model_visual_calibrated.png"
# )
# @show params

# ###
# ### Summary Plots
# ###
# using CairoMakie
# using LinearAlgebra

# ### Parameter convergence plot

# θ_mean = hcat(getproperty.(eki.iteration_summaries, :ensemble_mean)...)
# θθ_std_arr = sqrt.(hcat(diag.(getproperty.(eki.iteration_summaries, :ensemble_cov))...))
# N_param, N_iter = size(θ_mean)
# iter_range = 0:(N_iter-1)
# pnames = calibration.free_parameters.names

# n_cols = 3
# n_rows = Int(ceil(N_param / n_cols))
# ax_coords = [(i, j) for i = 1:n_rows, j = 1:n_cols]

# f = Figure(resolution = (500n_cols, 200n_rows))
# for (i, pname) in zip(1:N_param, pnames)
#     coords = ax_coords[i]
#     ax = Axis(f[coords...],
#         xlabel = "Iteration",
#         xticks = iter_range,
#         ylabel = string(pname))

#     ax.ylabelsize = 20

#     lines!(ax, iter_range, θ_mean[i, :])
#     band!(ax, iter_range, θ_mean[i, :] .+ θθ_std_arr[i, :], θ_mean[i, :] .- θθ_std_arr[i, :])
#     hlines!(ax, [true_parameters[pname]], color = :red)
# end

# save("perfect_model_calibration_parameter_convergence.png", f);

# ### Pairwise ensemble plots

# N_param, N_iter = size(θ_mean)
# for p1 = 1:N_param, p2 = 1:N_param
#     if p1 < p2
#         pname1, pname2 = pnames[[p1, p2]]

#         f = Figure()
#         axtop = Axis(f[1, 1])
#         axmain = Axis(f[2, 1],
#             xlabel = string(pname1),
#             ylabel = string(pname2))
#         axright = Axis(f[2, 2])
#         scatters = []
#         for iteration in [1, 2, 3, N_iter]
#             ensemble = transpose(eki.iteration_summaries[iteration].parameters[[p1, p2], :])
#             push!(scatters, scatter!(axmain, ensemble))
#             density!(axtop, ensemble[:, 1])
#             density!(axright, ensemble[:, 2], direction = :y)
#         end
#         vlines!(axmain, [true_parameters[p1]], color = :red)
#         vlines!(axtop, [true_parameters[p1]], color = :red)
#         hlines!(axmain, [true_parameters[p2]], color = :red, alpha = 0.6)
#         hlines!(axright, [true_parameters[p2]], color = :red, alpha = 0.6)
#         colsize!(f.layout, 1, Fixed(300))
#         colsize!(f.layout, 2, Fixed(200))
#         rowsize!(f.layout, 1, Fixed(200))
#         rowsize!(f.layout, 2, Fixed(300))
#         Legend(f[1, 2], scatters,
#             ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration $N_iter"],
#             position = :lb)
#         hidedecorations!(axtop, grid = false)
#         hidedecorations!(axright, grid = false)
#         # xlims!(axmain, 350, 1350)
#         # xlims!(axtop, 350, 1350)
#         # ylims!(axmain, 650, 1750)
#         # ylims!(axright, 650, 1750)
#         xlims!(axright, 0, 10)
#         ylims!(axtop, 0, 10)
#         save("eki_$(pname1)_$(pname2).png", f)
#     end
# end

# # Compare EKI result to true values

# weight_distances = [norm(θ̅(iter) - θ★) for iter = 1:iterations]
# output_distances = mapslices(norm, (forward_map(calibration, θ_mean)[:, 1:N_iter] .- y), dims = 1)

# f = Figure()
# lines(f[1, 1], 1:iterations, weight_distances, color = :red, linewidth = 2,
#     axis = (title = "Parameter distance",
#         xlabel = "Iteration",
#         ylabel = "|θ̅ₙ - θ⋆|",
#         yscale = log10))
# lines(f[1, 2], 1:iterations, output_distances, color = :blue, linewidth = 2,
#     axis = (title = "Output distance",
#         xlabel = "Iteration",
#         ylabel = "|G(θ̅ₙ) - y|",
#         yscale = log10))

# axislegend(ax3, position = :rt)
# save("error_convergence_summary.png", f);
