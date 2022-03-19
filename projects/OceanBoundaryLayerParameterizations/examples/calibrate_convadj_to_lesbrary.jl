# Calibrate convective adjustment closure parameters to LESbrary 2-day "free_convection" simulation

using OceanBoundaryLayerParameterizations
using OceanLearning
using OceanLearning.Transformations: Transformation
using LinearAlgebra, CairoMakie, DataDeps
using Oceananigans
using Oceananigans.Units

architecture = CPU()
Nensemble = 30
description = ""
Δt = 10.0

###
### Build an observation from "free convection" LESbrary simulation
###

data_path = datadep"two_day_suite_4m/free_convection_instantaneous_statistics.jld2" # Nz = 64
data_path_highres = datadep"two_day_suite_2m/free_convection_instantaneous_statistics.jld2" # Nz = 128

transformation = (b = Transformation(normalization=ZScore()),)

observation, observation_highres = SyntheticObservations.([data_path, data_path_highres]; 
                                    field_names=(:b,), 
                                    times=[3hours, 12hours, 48hours], 
                                    transformation, 
                                    regrid=(1, 1, 32)
                                    )

observations = [observation]
output_map = ConcatenatedOutputMap()

Γy = estimate_η_covariance(output_map, [observation, observation_highres])

f = Figure()
noise_var = diag(Γy)
lines(f[1,1], noise_var, 1:length(noise_var))
save("noise_var.pdf", f)

f = heatmap(Γy)
save("noise_cov.pdf", f)

f = Figure()
y = [observation_map(output_map, observation)...]
lines(f[1,1], y, 1:length(y))
save("obs_map.pdf", f)

###
### Build Inverse Problem
###

priors = (convective_κz = ScaledLogitNormal(bounds=(0.1, 2.1)),
          background_κz = ScaledLogitNormal(bounds=(0.0, 2.5e-3)))
free_parameters = FreeParameters(priors)

closure = ConvectiveAdjustmentVerticalDiffusivity()

simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, closure, Δt)
calibration = InverseProblem(observations, simulation, free_parameters; output_map = output_map)

function build_inverse_problem(Nensemble)
    simulation = lesbrary_ensemble_simulation(observations; Nensemble, architecture, closure, Δt)
    calibration = InverseProblem(observations, simulation, free_parameters; output_map)
    return calibration
end

calibration = build_inverse_problem(Nensemble)

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

directory = "calibrate_convadj_to_lesbrary/$(iterations)_iters_$(Nensemble)_particles_$(description)/$(noise_cov_name)/"
isdir(directory) || mkpath(directory)

plot_parameter_convergence!(eki, directory)
plot_pairwise_ensembles!(eki, directory)
plot_error_convergence!(eki, directory)

visualize!(calibration, eki.iteration_summaries[end].ensemble_mean;
    field_names = (:b,),
    directory = directory,
    filename = "realizations.pdf"
)

test = (convective_κz = 0.4, background_κz = 0.0002)
visualize!(calibration, test;
    field_names = (:b,),
    directory = directory,
    filename = "test.pdf"
)

###
### Visualize loss landscape
###

name = "Loss landscape"

# pvalues = Dict(
#     :convective_κz => collect(0.1:0.05:2.0),
#     :background_κz => collect(0.5e-4:0.5e-4:1e-3),
# )

pvalues = Dict(
    :convective_κz => collect(0.1:0.05:2.1),
    :background_κz => collect(0.5e-4:1e-4:2.5e-3),
)

p1 = pvalues[:convective_κz]
p2 = pvalues[:background_κz]

ni = length(p1)
nj = length(p2)

params = hcat([[p1[i], p2[j]] for i = 1:ni, j = 1:nj]...)
xc = params[1, :]
yc = params[2, :]

# build an `InverseProblem` that can accommodate `ni*nj` ensemble members
big_calibration = build_inverse_problem(ni*nj)

y = observation_map(big_calibration)

using FileIO

file = "calibrate_convadj_to_lesbrary/loss_landscape_$(description).jld2"

# @time G = forward_map(big_calibration, params)
G = load(file)["G"]

θglobalmin = NamedTuple((:convective_κz => 0.275, :background_κz => 0.000275))
visualize!(calibration, θglobalmin;
    field_names = (:b,),
    directory = directory,
    filename = "realizations_θglobalmin.pdf"
)

using OceanLearning.EnsembleKalmanInversions: eki_objective
Φs = [eki_objective(eki, params[:,j], G[:,j]) for j in 1:size(G, 2)]

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
            zlabel = zlabel,
            shading = true,
            shininess = 32.0,
            transparency = true
        )

        # CairoMakie.surface!(ax1, xc, yc, zc, colorscheme = :thermal)
        CairoMakie.surface!(ax1, xc, yc, zc)

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