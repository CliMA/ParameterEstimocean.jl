pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using OceanTurbulenceParameterEstimation, LinearAlgebra, CairoMakie
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity
using OceanTurbulenceParameterEstimation.EnsembleKalmanInversions: NaNResampler, FullEnsembleDistribution
using OceanBoundaryLayerParameterizations

examples_path = joinpath(pathof(OceanTurbulenceParameterEstimation), "../../examples")
include(joinpath(examples_path, "intro_to_inverse_problems.jl"))

###
### Pick a parameter set defined in `./utils/parameters.jl`
###

parameter_set = CATKEParametersRiDependent
closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64;), parameter_set.settings)

###
### Pick the secret `true_parameters`
###

true_parameters = (Cᵟu = 0.5, CᴷRiʷ = 1.0, Cᵂu★ = 2.0, CᵂwΔ = 1.0, Cᴷeʳ = 5.0, Cᵟc = 0.5, Cᴰ = 2.0, Cᴷc⁻ = 0.5, Cᴷe⁻ = 0.2, Cᴷcʳ = 3.0, Cᴸᵇ = 1.0, CᴷRiᶜ = 1.0, Cᴷuʳ = 4.0, Cᴷu⁻ = 0.8, Cᵟe = 0.5)
true_closure = closure_with_parameters(closure, true_parameters)

###
### Generate and load synthetic observations
###

kwargs = (tracers = (:b, :e), Δt = 10.0, stop_time = 1day)

# Note: if an output file of the same name already exist, `generate_synthetic_observations` will return the existing path and skip re-generating the data.
observ_path1 = generate_synthetic_observations("perfect_model_observation1"; Qᵘ = 3e-5, Qᵇ = 7e-9, f₀ = 1e-4, closure = true_closure, kwargs...)
observ_path2 = generate_synthetic_observations("perfect_model_observation2"; Qᵘ = -2e-5, Qᵇ = 3e-9, f₀ = 0, closure = true_closure, kwargs...)
observation1 = SyntheticObservations(observ_path1, field_names = (:b, :e, :u, :v), normalize = ZScore)
observation2 = SyntheticObservations(observ_path2, field_names = (:b, :e, :u, :v), normalize = ZScore)
observations = [observation1, observation2]

###
### Build Ensemble Simulation
###

ensemble_simulation, closure = build_ensemble_simulation(observations; Nensemble = 100, architecture = GPU())

###
### Build Inverse Problem
###

build_prior(name) = ConstrainedNormal(0.0, 1.0, bounds(name) .* 0.5...)
free_parameters = FreeParameters(named_tuple_map(names(parameter_set), build_prior))

# Pack everything into Inverse Problem `calibration`
track_times = Int.(floor.(range(1, stop = length(observations[1].times), length = 3)))
calibration = InverseProblem(observations, ensemble_simulation, free_parameters, output_map = ConcatenatedOutputMap(track_times));

###
### Run EKI
###

noise_covariance = 0.01
resampler = NaNResampler(; abort_fraction=0.5, distribution=FullEnsembleDistribution())

eki = EnsembleKalmanInversion(calibration; noise_covariance = noise_covariance, 
                                           resampler = resampler)
params = iterate!(eki; iterations = 5)

###
### Summary Plots
###

using CairoMakie
using LinearAlgebra

directory = "calibrate_catke_to_perfect_model/"
isdir(directory) || mkpath(directory)

visualize!(calibration, true_parameters;
    field_names = (:u, :v, :b, :e),
    directory = directory,
    filename = "perfect_model_visual_true_params.png"
)

visualize!(calibration, params;
    field_names = [:u, :v, :b, :e],
    directory = directory,
    filename = "perfect_model_visual_calibrated.png"
)
@show params

plot_parameter_convergence!(eki, directory, true_parameters, n_columns=3)
plot_pairwise_ensembles!(eki, directory, true_parameters)
plot_error_convergence!(eki, directory, true_parameters)