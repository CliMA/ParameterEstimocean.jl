# # Perfect CAKTE calibration with Ensemble Kalman Inversion

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, CairoMakie"
# ```

using OceanTurbulenceParameterEstimation, LinearAlgebra, CairoMakie
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity, MixingLength, SurfaceTKEFlux

examples_path = joinpath(pathof(OceanTurbulenceParameterEstimation), "..", "..", "examples")
include(joinpath(examples_path, "intro_to_inverse_problems.jl"))

mixing_length = MixingLength(Cᴬu=0.1, Cᴬc=0.1, Cᴬe=0.1, Cᴷuʳ=0.0, Cᴷcʳ=0.0, Cᴷeʳ=0.0)
catke = CATKEVerticalDiffusivity(mixing_length=mixing_length)
data_path = generate_synthetic_observations("catke", closure=catke, tracers=(:b, :e), Δt=10.0)
observations = SyntheticObservations(data_path, field_names=(:u, :v, :b, :e), normalize=ZScore)

ensemble_simulation, closure★ = build_ensemble_simulation(observations; Nensemble=50)

priors = (Cᴷu⁻ = lognormal_with_mean_std(0.01, 0.1),
          Cᴷc⁻ = lognormal_with_mean_std(0.01, 0.1),
          Cᴷe⁻ = lognormal_with_mean_std(0.01, 0.1),
          Cᴸᵇ = lognormal_with_mean_std(0.2, 0.1),
          Cᴰ = lognormal_with_mean_std(1.0, 0.5),
          CᵂwΔ = lognormal_with_mean_std(1.0, 0.2))

free_parameters = FreeParameters(priors)

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# # Ensemble Kalman Inversion
#
# Next, we construct an `EnsembleKalmanInversion` (EKI) object,
#
# The calibration is done here using Ensemble Kalman Inversion. For more information about the 
# algorithm refer to
# [EnsembleKalmanProcesses.jl documentation](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/ensemble_kalman_inversion/).

noise_variance = observation_map_variance_across_time(calibration)[1, :, 1] .+ 1e-5

eki = EnsembleKalmanInversion(calibration; noise_covariance = Matrix(Diagonal(noise_variance)))

# and perform few iterations to see if we can converge to the true parameter values.

iterate!(eki; iterations = 10)

# Last, we visualize the outputs of EKI calibration.

θ̅(iteration) = [eki.iteration_summaries[iteration].ensemble_mean...]
varθ(iteration) = eki.iteration_summaries[iteration].ensemble_var

weight_distances = [norm(θ̅(iter) - [θ★[1], θ★[2]]) for iter in 1:eki.iteration]
output_distances = [norm(forward_map(calibration, θ̅(iter))[:, 1] - y) for iter in 1:eki.iteration]
ensemble_variances = [varθ(iter) for iter in 1:eki.iteration]

f = Figure()

lines(f[1, 1], 1:eki.iteration, weight_distances, color = :red, linewidth = 2,
      axis = (title = "Parameter distance",
              xlabel = "Iteration",
              ylabel = "|θ̅ₙ - θ★|"))

lines(f[1, 2], 1:eki.iteration, output_distances, color = :blue, linewidth = 2,
      axis = (title = "Output distance",
              xlabel = "Iteration",
              ylabel = "|G(θ̅ₙ) - y|"))

ax3 = Axis(f[2, 1:2],
           title = "Parameter convergence",
           xlabel = "Iteration",
           ylabel = "Ensemble variance",
           yscale = log10)

for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances, i)
    lines!(ax3, 1:eki.iteration, ev / ev[1], label = String(pname), linewidth = 2)
end

axislegend(ax3, position = :rt)

save("summary_catke_eki.svg", f); nothing #hide

# ![](summary_catke_eki.svg)

# And also we plot the the distributions of the various model ensembles for few EKI iterations to see
# if and how well they converge to the true diffusivity values.

f = Figure()

axtop = Axis(f[1, 1])

axmain = Axis(f[2, 1],
              xlabel = "Cᴷu⁻ [m² s⁻¹]",
              ylabel = "Cᴷc⁻ [m² s⁻¹]")

axright = Axis(f[2, 2])
scatters = []

for iteration in [1, 2, 3, 11]
    ## Make parameter matrix
    parameters = eki.iteration_summaries[iteration].parameters
    Nensemble = length(parameters)
    Nparameters = length(first(parameters))
    parameter_ensemble_matrix = [parameters[i][j] for i=1:Nensemble, j=1:Nparameters]

    push!(scatters, scatter!(axmain, parameter_ensemble_matrix))
    density!(axtop, parameter_ensemble_matrix[:, 1])
    density!(axright, parameter_ensemble_matrix[:, 2], direction = :y)
end

vlines!(axmain, [θ★.Cᴷu⁻], color = :red)
vlines!(axtop, [θ★.Cᴷu⁻], color = :red)

hlines!(axmain, [θ★.Cᴷc⁻], color = :red)
hlines!(axright, [θ★.Cᴷc⁻], color = :red)

colsize!(f.layout, 1, Fixed(300))
colsize!(f.layout, 2, Fixed(200))
rowsize!(f.layout, 1, Fixed(200))
rowsize!(f.layout, 2, Fixed(300))

Legend(f[1, 2], scatters, ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration 10"],
       position = :lb)

hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)

xlims!(axmain, -0.25, 3.2)
xlims!(axtop, -0.25, 3.2)
ylims!(axmain, 5e-5, 35e-5)
ylims!(axright, 5e-5, 35e-5)

save("distributions_catke_eki.svg", f); nothing #hide

# ![](distributions_catke_eki.svg)
