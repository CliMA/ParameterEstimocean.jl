# Calibrate convective adjustment closure parameters to LESbrary 2-day "free_convection" simulation

using OceanBoundaryLayerParameterizations
using OceanLearning
using OceanLearning.Transformations: Transformation
using LinearAlgebra, CairoMakie, DataDeps
using Oceananigans
using Oceananigans.Units
using Plots
using PyPlot
using PlotlyJS
using PyPlot

architecture = CPU()
Nensemble = 25
forward_map_description = ""
Δt = 10.0

directory = "calibrate_convadj_to_lesbrary/$(iterations)_iters_$(Nensemble)_particles_$(forward_map_description)/$(noise_cov_name)/"
isdir(directory) || mkpath(directory)

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

Nobs = length(y)

# noise_cov_name = "noise_covariance_128_64_resolutions_to_32"
# Γy = estimate_η_covariance(output_map, [observation, observation_highres]) .+ Matrix(1e-10 * I, Nobs, Nobs)

noise_cov_name = "noise_covariance_01"
Γy = Matrix(0.01 * I, Nobs, Nobs)

begin
    f = CairoMakie.Figure(resolution = (500, 600), fontsize = 24)
    ax1 = Axis(f[1, 1], title = "y")
    ax2 = Axis(f[1, 2], title = "diag(Γy)")

    y = [observation_map(output_map, observation)...]
    lines!(ax1, y, 1:length(y), color=:purple, linewidth=4)

    noise_var = diag(Γy)
    lines!(ax2, noise_var, 1:length(noise_var), color=:purple, linewidth=4)

    save(joinpath(directory, "obs_and_noise_var.pdf"), f)
end

f = CairoMakie.heatmap(Γy)
save(joinpath(directory, "noise_cov_heatmap.pdf"), f)


###
### Build Inverse Problem
###

priors = (convective_κz = ScaledLogitNormal(bounds=(0.0, 2.0)),
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

unconstrained_prior1 = unconstrained_prior(priors[:convective_κz])
unconstrained_prior2 = unconstrained_prior(priors[:background_κz])

# n = 10
# uniform_unit_samples = collect(range(0.0; stop=1.0, length=n+2)[2:n+1])
# samples1 = quantile(unconstrained_prior1, uniform_unit_samples)
# samples2 = quantile(unconstrained_prior2, uniform_unit_samples)
# initial_ensemble = hcat([[s1, s2] for s1 in samples1, s2 in samples2]...)

iterations = 10
eki = EnsembleKalmanInversion(calibration; noise_covariance = Γy)
iterate!(eki; iterations = iterations)

###
### Summary Plots
###

plot_parameter_convergence!(eki, directory)
plot_pairwise_ensembles!(eki, directory)
plot_error_convergence!(eki, directory)

visualize!(calibration, eki.iteration_summaries[end].ensemble_mean;
    field_names = (:b,),
    directory = directory,
    filename = "realizations.pdf"
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

file = "calibrate_convadj_to_lesbrary/loss_landscape_$(forward_map_description).jld2"

# @time G = forward_map(big_calibration, params)
G = load(file)["G"]

θglobalmin = NamedTuple((:convective_κz => 0.275, :background_κz => 0.000275))
visualize!(calibration, θglobalmin;
    field_names = (:b,),
    directory = directory,
    filename = "realizations_θglobalmin.pdf"
)

using OceanLearning.EnsembleKalmanInversions: eki_objective
Φs = [eki_objective(eki, params[:,j], G[:,j]; constrained = true) for j in 1:size(G, 2)]

save(file, Dict("G" => G,
                noise_cov_name*"/Φ2" => getindex.(Φs, 2),
                noise_cov_name*"/Φ1" => getindex.(Φs, 1)))

function plot_contour(eki, xc, yc, zc, name, directory; zlabel="MSE loss", title="EKI Particle Traversal Over Loss Landscape", plot_minimizer=false, plot_scatters=true)

    # 2D contour plot with EKI particles superimposed
    begin
        f = CairoMakie.Figure()
        ax1 = Axis(f[1, 1],
            title = title,
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

        plot_scatters && arrows!(xs, ys, us, vs, arrowsize = 10, lengthscale = 0.3,
            arrowcolor = :yellow, linecolor = :yellow)

        legend_labels = Vector{String}([])
        scatters = []

        if plot_minimizer

            # Ignore all NaNs
            not_nan_indices = findall(.!isnan.(zc))
            xc_no_nans = xc[not_nan_indices]
            yc_no_nans = yc[not_nan_indices]
            zc_no_nans = zc[not_nan_indices]
        
            am = argmin(zc_no_nans)
            minimizing_params = [xc_no_nans[am] yc_no_nans[am]]
            push!(scatters, CairoMakie.scatter!(ax1, minimizing_params, marker = :x, markersize = 30))
            push!(legend_labels, "Global minimum")
        end

        if plot_scatters
            for (i, iteration) in enumerate([0, 1, iterations])
                ensemble = eki.iteration_summaries[iteration].parameters
                ensemble = [[particle[:convective_κz], particle[:background_κz]] for particle in ensemble]
                ensemble = transpose(hcat(ensemble...)) # N_ensemble x 2
                push!(scatters, CairoMakie.scatter!(ax1, ensemble))
            end
            legend_labels = vcat(legend_labels, ["Initial ensemble", "Iteration 1", "Iteration $(iterations)"])
        end

        if plot_minimizer || plot_scatters
            Legend(f[1, 2], scatters, legend_labels, position = :lb)
        end

        save(joinpath(directory, "loss_contour_$(name).pdf"), f)
    end

    # 3D loss landscape
    begin
        f = CairoMakie.Figure()
        ax1 = Axis3(f[1, 1],
            title = title,
            xlabel = "convective_κz",
            ylabel = "background_κz",
            zlabel = zlabel,
            shading = true,
            shininess = 32.0,
            transparency = true
        )

        CairoMakie.surface!(ax1, xc, yc, zc)
        save(joinpath(directory, "loss_landscape_$(name).pdf"), f)

        #######
        #######
        #######

        # layout = PlotlyJS.Layout(
        #     title=title,
        #     xtitle="convective_κz",
        #     ytitle="background_κz",
        #     autosize=false,
        #     scene_camera_eye=attr(x=-1.25, y=-1.25, z=1.25),
        #     width=250, height=250
        # )
        # p = PlotlyJS.plot(PlotlyJS.surface(
        #     x=collect(1:ni),
        #     y=collect(1:nj),
        #     z=reshape(zc, (ni,nj)),
        #     opacity=0.9,
        #     contours_z=attr(
        #         show=true,
        #         usecolormap=true,
        #         highlightcolor="limegreen",
        #         project_z=true
        #     )
        # ), layout)
        # PlotlyJS.savefig(p, joinpath(directory, "v3_loss_landscape_$(name).png"), width=1000, height=1000)

        #######
        #######
        #######

        # xgrid = reshape(xc, (ni,nj))
        # ygrid = reshape(yc, (ni,nj))
        # zgrid = reshape(zc, (ni,nj))
        # fig = PyPlot.figure("pyplot_surfaceplot",figsize=(10,10))
        # ax = fig.add_subplot(2,1,1,projection="3d")
        # ax.plot_surface(xgrid, ygrid, zgrid, rstride=2, cstride=2, edgecolors="k", cmap="autumn_r", alpha=0.8, linewidth=0.0)
        # PyPlot.xlabel("convective_κz")
        # PyPlot.ylabel("background_κz")
        # PyPlot.zlabel(zlabel)
        # PyPlot.title("Surface Plot")

        # ax.contourf(xgrid, ygrid, zgrid, 10, linewidth=3, cmap="autumn_r", linestyles="solid", offset=-1)

        # PyPlot.savefig(joinpath(directory, "v3_loss_landscape_$(name).pdf"))
        # PyPlot.close(fig)

        #######
        #######
        #######

        # ax.contourf(xgrid, ygrid, zgrid, 10, linewidth=3, cstride=1, rstride=1, cmap="autumn_r", linestyles="solid", offset=-1)
        # ax.contourf(xgrid, ygrid, zgrid, 10, linewidth=3, colors="k", linestyles="solid")
        
        # PyPlot.subplot(212)
        # ax = fig.add_subplot(2,1,2)
        # cp = PyPlot.contour(xgrid, ygrid, zgrid, colors="black")
        # ax.clabel(cp, inline=1, fontsize=10)
        # PyPlot.xlabel("convective_κz")
        # PyPlot.ylabel("background_κz")
        # PyPlot.title("Contour Plot")
        # PyPlot.tight_layout()
    
    end
end

G = load(file)["G"]
zc = [mapslices(norm, G .- y, dims = 1)...]

Φ1 = load(file)[noise_cov_name*"/Φ1"]
Φ2 = load(file)[noise_cov_name*"/Φ2"]

plot_contour(eki, xc, yc, zc, "norm", directory; zlabel = "|G(θ) - y|", title="Mean Square Error", plot_minimizer=true, plot_scatters=false)
plot_contour(eki, xc, yc, Φ1, "Φ1", directory; zlabel = "Φ1", plot_scatters=false, title="Φ1")
plot_contour(eki, xc, yc, Φ2, "Φ2", directory; zlabel = "Φ2", plot_scatters=false, title="Φ2")
plot_contour(eki, xc, yc, Φ1 .+ Φ2, "Φ", directory; zlabel = "Φ", plot_minimizer=true, plot_scatters=false, title="EKI Objective Function, Φ")

particles_overlaid_directory = joinpath(directory, "with_particles_overlaid")
isdir(particles_overlaid_directory) || mkpath(particles_overlaid_directory)
plot_contour(eki, xc, yc, zc, "norm", particles_overlaid_directory; zlabel = "|G(θ) - y|", title="Mean Square Error", plot_minimizer=true, plot_scatters=true)
plot_contour(eki, xc, yc, Φ1, "Φ1", particles_overlaid_directory; zlabel = "Φ1", plot_scatters=true, title="Φ1")
plot_contour(eki, xc, yc, Φ2, "Φ2", particles_overlaid_directory; zlabel = "Φ2", plot_scatters=true, title="Φ2")
plot_contour(eki, xc, yc, Φ1 .+ Φ2, "Φ", particles_overlaid_directory; zlabel = "Φ", plot_minimizer=true, plot_scatters=true, title="EKI Objective Function, Φ")