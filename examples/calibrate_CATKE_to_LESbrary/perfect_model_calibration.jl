pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using Plots, LinearAlgebra, Distributions, JLD2
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using OceanTurbulenceParameterEstimation

include("lesbrary_paths.jl")
include("parameters.jl")
include("visualize_profile_predictions.jl")

#####
##### Simulation parameters
#####

Nz = 32
Lz = 64
Δt = 10.0
N² = 1e-6
stop_time = 1day
save_interval = 1hours
ensemble_size = 100
generate_observations = true

#####
##### Generate synthetic observations
#####

parameter_set = CATKEParametersRiDependent
closure = closure_with_parameter_set(CATKEVerticalDiffusivity(Float64;), parameter_set)

true_parameters = (Cᵟu = 0.5, CᴷRiʷ = 1.0, Cᵂu★ = 2.0, CᵂwΔ = 1.0, Cᴷeʳ = 5.0, Cᵟc = 0.5, Cᴰ = 2.0, Cᴷc⁻ = 0.5, Cᴷe⁻ = 0.2, Cᴷcʳ = 3.0, Cᴸᵇ = 1.0, CᴷRiᶜ = 1.0, Cᴷuʳ = 4.0, Cᴷu⁻ = 1.2, Cᵟe = 0.5)
true_closure = closure_with_parameters(closure, true_parameters)

function generate_truth_data!(name; Qᵘ, Qᵇ, f₀)

    grid = RectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))

    model = HydrostaticFreeSurfaceModel(grid = grid,
        tracers = (:b, :e),
        buoyancy = BuoyancyTracer(),
        boundary_conditions = (; u = u_bcs, b = b_bcs),
        coriolis = FPlane(f = f₀),
        closure = true_closure)

    set!(model, b = (x, y, z) -> N² * z)

    simulation = Simulation(model; Δt, stop_time)

    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
        schedule = TimeInterval(save_interval),
        prefix = joinpath(@__DIR__, name),
        array_type = Array{Float64},
        field_slicer = nothing,
        force = true)

    run!(simulation)
end

experiment_name = "perfect_model_observation1"
data_path = joinpath(@__DIR__, experiment_name * ".jld2")

if generate_observations || !(isfile(data_path))
    generate_truth_data!(experiment_name; Qᵘ = 2e-5, Qᵇ = 2e-8, f₀ = 1e-4)
end

experiment_name2 = "perfect_model_observation2"
data_path2 = joinpath(@__DIR__, experiment_name2 * ".jld2")

if generate_observations || !(isfile(data_path2))
    generate_truth_data!(experiment_name2; Qᵘ = -5e-5, Qᵇ = 0, f₀ = -1e-4)
end

#####
##### Load truth data as observations
#####

# observations = SixDaySuite("/Users/adelinehillier/.julia/dev")

observation1 = OneDimensionalTimeSeries(data_path, field_names = (:b, :e, :u, :v), normalize = ZScore)
observation2 = OneDimensionalTimeSeries(data_path2, field_names = (:b, :e, :u, :v), normalize = ZScore)
observations = [observation1, observation2]

Nx = ensemble_size
Ny = length(observations)
column_ensemble_size = ColumnEnsembleSize(Nz = Nz, ensemble = (Nx, Ny), Hz = 1)

#####
##### Set up ensemble model
#####

ensemble_grid = RectilinearGrid(size = column_ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
closure_ensemble = [closure for i = 1:Nx, j = 1:Ny]

function get_parameter(observation, parameter_path)
    file = jldopen(observation.path)
    parameter = file[parameter_path]
    close(file)
    return parameter
end

get_Qᵇ(observation) = get_parameter(observation, "timeseries/b/serialized/boundary_conditions").top.condition
get_Qᵘ(observation) = get_parameter(observation, "timeseries/u/serialized/boundary_conditions").top.condition
get_f₀(observation) = get_parameter(observation, "coriolis/f")

Qᵇ_ensemble = [get_Qᵇ(observations[j]) for i = 1:Nx, j = 1:Ny]
Qᵘ_ensemble = [get_Qᵘ(observations[j]) for i = 1:Nx, j = 1:Ny]

ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ_ensemble), bottom = GradientBoundaryCondition(N²))
ensemble_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ_ensemble))
coriolis_ensemble = [FPlane(f = get_f₀(observations[j])) for i = 1:Nx, j = 1:Ny]

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
    tracers = (:b, :e),
    buoyancy = BuoyancyTracer(),
    boundary_conditions = (; u = ensemble_u_bcs, b = ensemble_b_bcs),
    coriolis = coriolis_ensemble,
    closure = closure_ensemble)

set!(ensemble_model, b = (x, y, z) -> N² * z)

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

#####
##### Build free parameters
#####

free_parameter_names = keys(parameter_set.defaults)
free_parameter_means = collect(values(parameter_set.defaults))
priors = NamedTuple(pname => ConstrainedNormal(0.0, 1.0, bounds(pname) .* 0.5...) for pname in free_parameter_names)

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters);

θ★ = collect(values(true_parameters))
x = forward_map(calibration, θ★)[:, 1:1]
y = observation_map(calibration)
@show x == y

visualize!(calibration, θ★;
    field_names = [:u, :v, :b, :e],
    directory = @__DIR__,
    filename = "perfect_model_visual.png"
)

# #####
# ##### Calibrate
# #####

iterations = 5
eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)
params = iterate!(eki; iterations = iterations)

visualize!(calibration, params;
    field_names = [:u, :v, :b, :e],
    directory = @__DIR__,
    filename = "perfect_model_visual_calibrated.png"
)
@show params

###
### Summary Plots
###

using CairoMakie
using LinearAlgebra

# Parameter convergence plot

θ_mean = hcat(getproperty.(eki.iteration_summaries, :ensemble_mean)...)
θθ_std_arr = sqrt.(hcat(diag.(getproperty.(eki.iteration_summaries, :ensemble_cov))...))
N_param, N_iter = size(θ_mean)
iter_range = 0:(N_iter-1)
pnames = calibration.free_parameters.names

n_cols = 3
n_rows = Int(ceil(N_param / n_cols))
ax_coords = [(i, j) for i = 1:n_rows, j = 1:n_cols]

f = Figure(resolution = (500n_cols, 200n_rows))
for (i, pname) in zip(1:N_param, pnames)
    coords = ax_coords[i]
    ax = Axis(f[coords...],
        xlabel = "Iteration",
        xticks = iter_range,
        ylabel = string(pname))

    ax.ylabelsize = 20

    lines!(ax, iter_range, θ_mean[i, :])
    band!(ax, iter_range, θ_mean[i, :] .+ θθ_std_arr[i, :], θ_mean[i, :] .- θθ_std_arr[i, :])
    hlines!(ax, [true_parameters[pname]], color = :red)
end

save("perfect_model_calibration_parameter_convergence.png", f);

# Pairwise ensemble plots

N_param, N_iter = size(θ_mean)
for p1 = 1:N_param, p2 = 1:N_param
    if p1 < p2
        pname1, pname2 = pnames[[p1, p2]]

        f = Figure()
        axtop = Axis(f[1, 1])
        axmain = Axis(f[2, 1],
            xlabel = string(pname1),
            ylabel = string(pname2))
        axright = Axis(f[2, 2])
        scatters = []
        for iteration in [1, 2, 3, N_iter]
            ensemble = transpose(eki.iteration_summaries[iteration].parameters[[p1, p2], :])
            push!(scatters, scatter!(axmain, ensemble))
            density!(axtop, ensemble[:, 1])
            density!(axright, ensemble[:, 2], direction = :y)
        end
        vlines!(axmain, [true_parameters[p1]], color = :red)
        vlines!(axtop, [true_parameters[p1]], color = :red)
        hlines!(axmain, [true_parameters[p2]], color = :red, alpha = 0.6)
        hlines!(axright, [true_parameters[p2]], color = :red, alpha = 0.6)
        colsize!(f.layout, 1, Fixed(300))
        colsize!(f.layout, 2, Fixed(200))
        rowsize!(f.layout, 1, Fixed(200))
        rowsize!(f.layout, 2, Fixed(300))
        Legend(f[1, 2], scatters,
            ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration $N_iter"],
            position = :lb)
        hidedecorations!(axtop, grid = false)
        hidedecorations!(axright, grid = false)
        # xlims!(axmain, 350, 1350)
        # xlims!(axtop, 350, 1350)
        # ylims!(axmain, 650, 1750)
        # ylims!(axright, 650, 1750)
        xlims!(axright, 0, 10)
        ylims!(axtop, 0, 10)
        save("eki_$(pname1)_$(pname2).png", f)
    end
end

# Compare EKI result to true values

weight_distances = [norm(θ̅(iter) - θ★) for iter = 1:iterations]
output_distances = mapslices(norm, (forward_map(calibration, θ_mean)[:, 1:N_iter] .- y), dims = 1)

f = Figure()
lines(f[1, 1], 1:iterations, weight_distances, color = :red, linewidth = 2,
    axis = (title = "Parameter distance",
        xlabel = "Iteration",
        ylabel = "|θ̅ₙ - θ⋆|",
        yscale = log10))
lines(f[1, 2], 1:iterations, output_distances, color = :blue, linewidth = 2,
    axis = (title = "Output distance",
        xlabel = "Iteration",
        ylabel = "|G(θ̅ₙ) - y|",
        yscale = log10))

for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances, i)
end

axislegend(ax3, position = :rt)
save("error_convergence_summary.png", f);
