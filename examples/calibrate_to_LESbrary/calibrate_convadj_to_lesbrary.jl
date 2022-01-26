# Calibrate convective adjustment closure parameters to LESbrary 2-day "free_convection" simulation

using OceanTurbulenceParameterEstimation, LinearAlgebra, CairoMakie, DataDeps
using Oceananigans.Units

include("utils/eki_visuals.jl")

###
### Build an observation from "free convection" LESbrary simulation
###

# observations = SyntheticObservationsBatch(suite; first_iteration = 13, last_iteration = nothing, normalize = ZScore, Nz = 64)

data_path = datadep"two_day_suite_4m/free_convection_instantaneous_statistics.jld2" # Nz = 64
data_path_highres = datadep"two_day_suite_2m/free_convection_instantaneous_statistics.jld2" # Nz = 128

field_names = (:b,)
observations = SyntheticObservations(data_path; field_names, normalize=ZScore, regrid_size=(1, 1, 64))


times = [2hours, 24hours, 36hours, 48hours]
field_names = (:b,)
observations = SyntheticObservations(data_path; field_names, times, normalize=ZScore, regrid_size=(1, 1, 64))
observation_highres = SyntheticObservations(data_path_highres; field_names, times, normalize=ZScore, regrid_size=(1, 1, 64))

## Specify an output map that tracks 2 uniformly spaced time steps (Ignores the initial condition.)
# track_times = Int.(floor.(range(1, stop = lastindex(observations[1].times), length = 3)))
output_map = ConcatenatedOutputMap()

function estimate_obs_covariance(data_paths)
    obs_maps = []
    for data_path in data_paths
        temp_observation = SyntheticObservations(data_path_highres; field_names, times, normalize=ZScore, regrid_size=(1, 1, 64))
        push!(obs_maps, observation_map(output_map, temp_observation))
    end
    # obs_maps = hcat(obs_maps...)
    return obs_maps
end

estimate_obs_covariance([data_path, data_path_highres])

###
### Make synthetic observations to approximate noise covariance matrix
###

closure = ConvectiveAdjustmentVerticalDiffusivity()

kwargs = (tracers = (:b,), Δt = 10.0, stop_time = 2days)

approx_closure = closure_with_parameters(closure, true_parameters)

# Note: if an output file of the same name already exist, `generate_synthetic_observations` will return the existing path and skip re-generating the data.
observ_path_high_res = generate_synthetic_observations("perfect_model_observation_high_res"; Qᵘ = 3e-5, Qᵇ = 7e-9, f₀ = 1e-4, closure = approx_closure, kwargs...)
observation1 = SyntheticObservations(observ_path1, field_names = (:b, :e, :u, :v), normalize = ZScore)
observations = [observation1, observation2]

Nz = 32, Lz = 64

###
### Build Inverse Problem
###

architecture = CPU()
ensemble_size = 50

priors = (
    convective_κz = ConstrainedNormal(0.0, 1.0, 0.1, 1.0),
    background_κz = ConstrainedNormal(0.0, 1.0, 0.0, 10e-4)
)

free_parameters = FreeParameters(priors)

description = "Nz_64_3times"

function build_inverse_problem(observations, ensemble_size)

    simulation = ensemble_column_model_simulation(observations;
                                                  Nensemble = 30,
                                                  architecture = CPU(),
                                                  tracers = (:b, :e),
                                                  closure = catke)

    # `ensemble_column_model_simulation` sets up `simulation`
    # with a `FluxBoundaryCondition` array initialized to 0 and a default
    # time-step. We modify these for our particular problem,

    simulation.Δt = 20.0

    Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
    Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
    N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

    for (case, obs) in enumerate(observations)
        view(Qᵘ, case, :) .= obs.metadata.parameters.momentum_flux
        view(Qᵇ, case, :) .= obs.metadata.parameters.buoyancy_flux
        view(N², case, :) .= obs.metadata.parameters.N²_deep
    end

    ensemble_model = OneDimensionalEnsembleModel(observations;
        architecture = architecture,
        ensemble_size = ensemble_size,
        closure = closure)
    ensemble_simulation = Simulation(ensemble_model; Δt = 10seconds)
    return InverseProblem(observations, ensemble_simulation, free_parameters; output_map = output_map)
end

calibration = build_inverse_problem(observations, ensemble_size)

###
### Run EKI
###

iterations = 3
noise_cov_name = "noise_covariance_1e-3"
eki = EnsembleKalmanInversion(calibration; noise_covariance = 0.001)
iterate!(eki; iterations = iterations)

###
### Summary Plots
###

directory = "calibrate_convadj_to_lesbrary/$(iterations)_iters_$(ensemble_size)_particles_$(description)/$(noise_cov_name)/"
isdir(directory) || mkpath(directory)

plot_parameter_convergence!(eki, directory)
plot_pairwise_ensembles!(eki, directory)
plot_error_convergence!(eki, directory)

include("./utils/visualize_profile_predictions.jl")
visualize!(calibration, ensemble_means(eki)[end];
    field_names = (:b,),
    directory = directory,
    filename = "realizations.pdf"
)

θglobalmin = NamedTuple((:convective_κz => 0.275, :background_κz => 0.000275))
visualize!(calibration, θglobalmin;
    field_names = (:b,),
    directory = directory,
    filename = "realizations_θglobalmin.pdf"
)

###
### Visualize loss landscape
###

name = "Loss landscape"

pvalues = Dict(
    :convective_κz => collect(0.075:0.025:1.025),
    :background_κz => collect(0e-4:0.5e-4:10e-4),
)

p1 = pvalues[:convective_κz]
p2 = pvalues[:background_κz]

params = hcat([[p1[i], p2[j]] for i = 1:length(p1), j = 1:length(p2)]...)
xc = params[1, :]
yc = params[2, :]

# build an `InverseProblem` that can accommodate `ni*nj` ensemble members 
big_calibration = build_inverse_problem(observations, ni * nj)

y = observation_map(big_calibration)

using FileIO

file = "calibrate_convadj_to_lesbrary/loss_landscape_$(description).jld2"

@time G = forward_map(big_calibration, params)
# G = load(file)["G"]

using OceanTurbulenceParameterEstimation.EnsembleKalmanInversions: Φ
Φs = [Φ(eki, params[:,j], G[:,j]) for j in 1:size(G, 2)]

save(file, Dict("G" => G,
                noise_cov_name*"/Φ2" => getindex.(Φs, 2),
                noise_cov_name*"/Φ1" => getindex.(Φs, 1)))

function plot_contour(eki, xc, yc, zc, name, directory; zlabel = "MSE loss")

    # 2D contour plot with EKI particles superimposed
    begin
        f = Figure()
        ax1 = Axis(f[1, 1],
            title = "EKI Particle Traversal Over Loss Landscape",
            xlabel = "convective_κz",
            ylabel = "background_κz")

        co = CairoMakie.contourf!(ax1, xc, yc, zc, levels = 50, colormap = :default)

        cvt(iter) = hcat(collect.(eki.iteration_summaries[iter].parameters)...)
        diffc = cvt(2) .- cvt(1)
        diff_mag = mapslices(norm, diffc, dims = 1)
        us = diffc[1, :]
        vs = diffc[2, :]
        xs = cvt(1)[1, :]
        ys = cvt(1)[2, :]

        arrows!(xs, ys, us, vs, arrowsize = 10, lengthscale = 0.3,
            arrowcolor = :yellow, linecolor = :yellow)

        am = argmin(zc)
        minimizing_params = [xc[am] yc[am]]

        scatters = [scatter!(ax1, minimizing_params, marker = :x, markersize = 30)]
        for (i, iteration) in enumerate([1, 2, iterations])
            ensemble = eki.iteration_summaries[iteration].parameters
            ensemble = [[particle[:convective_κz], particle[:background_κz]] for particle in ensemble]
            ensemble = transpose(hcat(ensemble...)) # N_ensemble x 2
            push!(scatters, scatter!(ax1, ensemble))
        end
        Legend(f[1, 2], scatters,
            ["Global minimum", "Initial ensemble", "Iteration 1", "Iteration $(iterations)"],
            position = :lb)

        save(joinpath(directory, "loss_contour_$(name).pdf"), f)
    end

    # 3D loss landscape
    begin
        f = Figure()
        ax1 = Axis3(f[1, 1],
            title = "Loss Landscape",
            xlabel = "convective_κz",
            ylabel = "background_κz",
            zlabel = zlabel
        )

        CairoMakie.surface!(ax1, xc, yc, zc, colorscheme = :thermal)

        save(joinpath(directory, "loss_landscape_$(name).pdf"), f)
    end
end

G = load(file)["G"]
zc = [mapslices(norm, G .- y, dims = 1)...]

Φ1 = load(file)[noise_cov_name*"/Φ1"]
Φ2 = load(file)[noise_cov_name*"/Φ2"]

plot_contour(eki, xc, yc, zc, "norm", directory; zlabel = "|G(θ) - y|")
plot_contour(eki, xc, yc, Φ1, "Φ1", directory; zlabel = "Φ1")
plot_contour(eki, xc, yc, Φ2, "Φ2", directory; zlabel = "Φ2")
plot_contour(eki, xc, yc, Φ1 .+ Φ2, "Φ", directory; zlabel = "Φ") 