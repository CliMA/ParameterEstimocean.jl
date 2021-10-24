pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Oceananigans
using Plots, LinearAlgebra, Distributions
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using OceanTurbulenceParameterEstimation

#####
##### Parameters
#####

Nz = 32
Lz = 64
Qᵇ = 1e-8
Qᵘ = -1e-5
Δt = 10.0
f₀ = 1e-4
N² = 1e-6
stop_time = 10hour
save_interval = 1hour
experiment_name = "convective_adjustment_example"
data_path = experiment_name * ".jld2"
ensemble_size = 50
generate_observations = false

free_parameters = (a = 12, d = 7)

# "True" parameters to be estimated by calibration
convective_κz = 1.0
convective_νz = 0.9
background_κz = 1e-4
background_νz = 1e-5

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

observations = OneDimensionalTimeSeries(data_path, field_names=(:b,), normalize=ZScore)

observations = [observations, observations]

#####
##### Set up ensemble model
#####

column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(ensemble_size, length(observations)), Hz=1)
ensemble_grid = RegularRectilinearGrid(size=column_ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
closure_ensemble = [ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz, convective_νz, background_νz) for i = 1:ensemble_grid.Nx, j = 1:ensemble_grid.Ny]
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
    convective_κz = lognormal_with_mean_std(0.3, 0.5),
    background_κz = lognormal_with_mean_std(2.5e-4, 0.25e-4),
)

θ★ = [convective_κz, background_κz]

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

v = observation_map_variance_across_time(calibration)[1,:,:]

# Assert that G(θ*) ≈ y
x = forward_map(calibration, θ★)
y = observation_map(calibration)
@show x == y

#####
##### Run calibration with EKI
#####

iterations = 10
noise_variance = observation_map_variance_across_time(calibration)[1,:,1] .+ 1e-5
eki = EnsembleKalmanInversion(calibration; noise_covariance=Matrix(Diagonal(noise_variance)));
params = iterate!(eki; iterations = iterations)

# eki = EnsembleKalmanInversion(calibration; noise_covariance=1e-2);
# params = iterate!(eki; iterations = iterations)

#####
##### Study EKI result
#####

output_size = length(x)
indices = 1:output_size
Nx, Ny, Nz = size(calibration.time_series_collector.grid)
Nt = (Int(stop_time / save_interval) + 1)
n = Ny*Nz

p = plot(size=(500,1000))
for t = 0:Nt-1
    range = (t*n+1):((t+1)*n)
    plot!(x[range], range, color=:red, legend=false)
    # plot!(y[range], range)
    plot!(0.1 .* v[range], range, color = :green)
end
savefig(p, "output_with_variance.png")
display(p)

@show params


###
### Summary plots
###

using CairoMakie

θ̅(iteration) = [eki.iteration_summaries[iteration].ensemble_mean...]
varθ(iteration) = eki.iteration_summaries[iteration].ensemble_variance

weight_distances = [norm(θ̅(iter) - θ★) for iter in 1:iterations]
output_distances = [norm(forward_map(calibration, θ̅(iter))[:, 1] - y) for iter in 1:iterations]
ensemble_variances = [varθ(iter) for iter in 1:iterations]

x = 1:iterations
f = Figure()
lines(f[1, 1], x, weight_distances, color = :red,
            axis = (title = "Parameter distance", xlabel = "Iteration, n", ylabel="|θ̅ₙ - θ⋆|"))
lines(f[1, 2], x, output_distances, color = :blue,
            axis = (title = "Output distance", xlabel = "Iteration, n", ylabel="|G(θ̅ₙ) - y|"))
ax3 = Axis(f[2, 1:2], title = "Parameter convergence", xlabel = "Iteration, n", ylabel="Ensemble variance")
for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances,i)
    lines!(ax3, 1:iterations, ev / ev[1], label=String(pname))
end
axislegend(ax3, position = :rt)
save("summary_makie.png", f)

###
### Plot ensemble density with time
###

f = Figure()
axtop = Axis(f[1, 1])
axmain = Axis(f[2, 1], xlabel = "convective_κz", ylabel = "background_κz")
axright = Axis(f[2, 2])
s = eki.iteration_summaries
scatters = []
for i in [1,2,3,10]
    ensemble = transpose(s[i].parameters)
    push!(scatters, CairoMakie.scatter!(axmain, ensemble))
    CairoMakie.density!(axtop, ensemble[:, 1])
    CairoMakie.density!(axright, ensemble[:, 2], direction = :y)
end
vlines!(axmain, [convective_κz], color=:red)
vlines!(axtop, [convective_κz], color=:red)
hlines!(axmain, [background_κz], color=:red)
hlines!(axright, [background_κz], color=:red)
colsize!(f.layout, 1, Fixed(300))
colsize!(f.layout, 2, Fixed(200))
rowsize!(f.layout, 1, Fixed(200))
rowsize!(f.layout, 2, Fixed(300))
leg = Legend(f[1, 2], scatters, ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration 10"], position = :lb)
hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)
save("distributions_makie.png", f)