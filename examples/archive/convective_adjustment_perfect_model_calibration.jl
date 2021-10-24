pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Distributions
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using OceanTurbulenceParameterEstimation
using Plots, LinearAlgebra
using LaTeXStrings

#####
##### Parameters
#####

Nz = 16
Lz = 64
Qᵇ = 1e-8
Qᵘ = -1e-5
Δt = 10.0
f₀ = 1e-4
N² = 1e-5
stop_time = 8hours
save_interval = 1hour
experiment_name = "convective_adjustment"
data_path = experiment_name * ".jld2"
ensemble_size = 100
generate_observations = false

# "True" parameters to be estimated by calibration
convective_κz = 1.0
convective_νz = 0.9
background_κz = 1e-4
background_νz = 1e-5

θ★ = [convective_κz; background_κz; convective_νz; background_νz]

#####
##### Generate synthetic observations
#####

if generate_observations || !(isfile(data_path))
    grid = RegularRectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
    closure = ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz, convective_νz, background_νz)
    coriolis = FPlane(f=f₀)
                                          
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))
    
    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (; u=u_bcs, b=b_bcs),
                                        coriolis = coriolis,
                                        closure = closure)
                                        
    set!(model, b = (x, y, z) -> N² * z)
    
    simulation = Simulation(model; Δt, stop_time)
    
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = TimeInterval(save_interval),
                                                          prefix = experiment_name,
                                                          array_type = Array{Float64},
                                                          field_slicer = nothing,
                                                          force = true)
    
    run!(simulation)
end

#####
##### Load truth data as observations
#####

data_path = experiment_name * ".jld2"

observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :b), normalize=ZScore)

# observations = [observations, observations]

#####
##### Set up ensemble model
#####

column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(ensemble_size, 1), Hz=1)
ensemble_grid = RegularRectilinearGrid(size=column_ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
closure_ensemble = [ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz) for i = 1:ensemble_grid.Nx, j = 1:ensemble_grid.Ny]
coriolis_ensemble = [FPlane(f=f₀) for i = 1:ensemble_grid.Nx, j = 1:ensemble_grid.Ny]

ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))
ensemble_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = :b,
                                             buoyancy = BuoyancyTracer(),
                                             boundary_conditions = (; u=ensemble_u_bcs, b=ensemble_b_bcs),
                                             coriolis = coriolis_ensemble,
                                             closure = closure_ensemble)

set!(ensemble_model, b = (x, y, z) -> N² * z)

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

pop!(ensemble_simulation.diagnostics, :nan_checker)

#####
##### Build free parameters
#####

priors = (
    convective_κz = lognormal_with_mean_std(1.2, 0.5),
    background_κz = lognormal_with_mean_std(1e-4, 1e-4),
    convective_νz = lognormal_with_mean_std(1.2, 0.5),
    background_νz = lognormal_with_mean_std(1e-5, 1e-5),
)

# priors = (
#     convective_κz = LogNormal(0, 0.7),
#     background_κz = LogNormal(0, 0.7),
#     convective_νz = LogNormal(0, 0.7),
#     background_νz = LogNormal(0, 0.7)
# )

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# forward_map(calibration, θ★)
x = forward_map(calibration, [θ★ for _ in 1:ensemble_size])[:, 1:1]
y = observation_map(calibration)

# Assert that G(θ*) ≈ y
@show x == y

# iterations = 5
# eki = EnsembleKalmanInversion(calibration; noise_covariance=1e-2)
# params = iterate!(eki; iterations = iterations)

# function noise_heatmap(calibration, iterations)
#     vrange = vrange=0.40:0.2:0.90
#     nlrange=-2.5:0.2:0.5
#     Γθs = collect(vrange)
#     Γys = 10 .^ collect(nlrange)
#     dist_from_true_params = zeros((length(Γθs), length(Γys)))
#     counter = 1
#     countermax = length(Γθs)*length(Γys)
#     for i in 1:length(Γθs)
#         for j in 1:length(Γys)
#             println("progress $(counter)/$(countermax)")
#             Γθ = Γθs[i]
#             Γy = Γys[j]
#             eki = EnsembleKalmanInversion(calibration; noise_covariance=Γys)
#             params = iterate!(eki; iterations = iterations)

#             dist_from_true_params[i, j] = norm([eki.iteration_summaries[end].ensemble_mean...] - θ★)
#             counter += 1
#         end
#     end
#     p = Plots.heatmap(Γys, Γθs, losses, xlabel=L"\Gamma_y", ylabel=L"\Gamma_\theta", size=(250,250), yscale=:log10)
#     Plots.savefig(p, joinpath(directory, "GammaHeatmap.png"))
#     Plots.savefig(p, joinpath(directory, "GammaHeatmap.pdf"))
#     v = Γθs[argmin(losses)[1]]
#     nl = Γys[argmin(losses)[2]]
#     println("loss-minimizing Γθ: $(v)")
#     println("loss-minimizing log10(Γy): $(log10(nl))")
#     return v, nl
# end

function plot_weight_distance_vs_noise_variance(ip, iterations, log_Γy_range = -2.5:0.1:0.5)
    Γys = 10 .^ collect(log_Γy_range)
    dist_from_true_params = []
    for Γy in Γys
        eki = EnsembleKalmanInversion(ip; noise_covariance=Γy)
        params = iterate!(eki; iterations = iterations)
        push!(dist_from_true_params, norm([eki.iteration_summaries[end].ensemble_mean...] - θ★))
    end
    p = Plots.plot(collect(log_Γy_range), dist_from_true_params)
    Plots.savefig(p, joinpath(pwd(), "noise_variance.png"))
    Plots.savefig(p, joinpath(pwd(), "noise_variance.pdf"))
    nl = Γys[argmin(dist_from_true_params)]
    println("loss-minimizing log10(Γy): $(log10(nl))")
    return nl
end

nl = plot_weight_distance_vs_noise_variance(calibration, 10, -2.5:0.05:0.5)

# @show params
# y = eki.mapped_observations

# weight_distances = [norm([eki.iteration_summaries[i].ensemble_mean...] - θ★) for i in 1:iterations]
# plot(1:iterations, weight_distances)