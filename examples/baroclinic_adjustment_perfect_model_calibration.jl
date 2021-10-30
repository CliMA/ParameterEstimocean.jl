pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Distributions
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Oceananigans: fields
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization
using OceanTurbulenceParameterEstimation

#####
##### Parameters
#####


# Domain
Lx = 500kilometers  # east-west extent [m]
Ly = 1000kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

Nx = 1
Ny = 64
Nz = 16

architecture = CPU()

stop_time = 10days
Δt = 15seconds
save_interval = 1days
experiment_name = "baroclinic_adjustment"
data_path = experiment_name * ".jld2"
ensemble_size = 10
generate_observations = false

# "True" parameters to be estimated by calibration
κ_skew = 1000.0       # [m² s⁻¹] skew diffusivity
κ_symmetric = 900.0  # [m² s⁻¹] symmetric diffusivity

Δx, Δy, Δz = Lx/Nx, Ly/Ny, Lz/Nz

𝒜 = Δz/Δx # Grid cell aspect ratio.

κh = 0.1    # [m² s⁻¹] horizontal diffusivity
νh = 0.1    # [m² s⁻¹] horizontal viscosity
κz = 𝒜 * κh # [m² s⁻¹] vertical diffusivity
νz = 𝒜 * νh # [m² s⁻¹] vertical viscosity

θ★ = [κ_skew, κ_symmetric]

diffusive_closure = AnisotropicDiffusivity(νh = νh,
                                           νz = νz,
                                           κh = κh,
                                           κz = κz,
                                           time_discretization = VerticallyImplicitTimeDiscretization())

convective_adjustment = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1.0,
                                                                convective_νz = 0.0)

gerdes_koberle_willebrand_tapering = Oceananigans.TurbulenceClosures.FluxTapering(1e-1)

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_skew,
                                                                κ_symmetric = κ_symmetric,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
                                        
closures = (diffusive_closure, convective_adjustment, gent_mcwilliams_diffusivity)

closures = gent_mcwilliams_diffusivity


#####
##### Generate synthetic observations
#####

if generate_observations || !(isfile(data_path))
    grid = RegularRectilinearGrid(topology = (Flat, Bounded, Bounded), 
                                  size = (Ny, Nz), 
                                  y = (-Ly/2, Ly/2),
                                  z = (-Lz, 0),
                                  halo = (3, 3))

    coriolis = BetaPlane(latitude=-45)
    
    model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                        grid = grid,
                                        coriolis = coriolis,
                                        buoyancy = BuoyancyTracer(),
                                        closure = closures,
                                        tracers = (:b, :c),
                                        momentum_advection = WENO5(),
                                        tracer_advection = WENO5(),
                                        free_surface = ExplicitFreeSurface())
    
    @info "Built $model."

    #####
    ##### Initial conditions
    #####

    """
    Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

    For example:

    y < y₀           => ramp = 0
    y₀ < y < y₀ + Δy => ramp = y / Δy
    y > y₀ + Δy      => ramp = 1
    """
    ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

    # Parameters
    N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
    M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

    Δy = 50kilometers
    Δz = 50

    Δc = 2Δy
    Δb = Δy * M²
    ϵb = 1e-2 * Δb # noise amplitude

    bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy)
    cᵢ(x, y, z) = exp(-y^2 / 2Δc^2) * exp(-(z + Lz/2)^2 / (2*Δz^2))

    set!(model, b=bᵢ, c=cᵢ)
    
    wall_clock = [time_ns()]
    
    function print_progress(sim)
        @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.8e, %6.8e, %6.8e) m/s\n",
                100 * (sim.model.clock.time / sim.stop_time),
                sim.model.clock.iteration,
                prettytime(sim.model.clock.time),
                prettytime(1e-9 * (time_ns() - wall_clock[1])),
                maximum(abs, sim.model.velocities.u),
                maximum(abs, sim.model.velocities.v),
                maximum(abs, sim.model.velocities.w))
    
        wall_clock[1] = time_ns()
        
        return nothing
    end
    
    simulation = Simulation(model, Δt=Δt, stop_time=stop_time, progress=print_progress, iteration_interval=1000)
    
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = TimeInterval(save_interval),
                                                          prefix = experiment_name,
                                                          array_type = Array{Float64},
                                                          field_slicer = nothing,
                                                          force = true)
    
    run!(simulation)
end

#=

#####
##### Visualize
#####
using GLMakie

fig = Figure(resolution = (1400, 700))

filepath = "baroclinic_adjustment.jld2"

ut = FieldTimeSeries(filepath, "u")
bt = FieldTimeSeries(filepath, "b")
ct = FieldTimeSeries(filepath, "c")

grid = RegularRectilinearGrid(topology = (Periodic, Bounded, Bounded), 
                                  size = (Nx, Ny, Nz), 
                                  x = (0, Lx),
                                  y = (-Ly/2, Ly/2),
                                  z = (-Lz, 0),
                                  halo = (3, 3, 3))

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes((Center, Center, Center), grid)

#####
##### Plot buoyancy...
#####

times = bt.times
Nt = length(times)

un(n) = interior(ut[n])[1, :, :]
bn(n) = interior(bt[n])[1, :, :]
cn(n) = interior(ct[n])[1, :, :]

@show min_c = 0
@show max_c = 1
@show max_u = maximum(abs, un(Nt))
min_u = - max_u

n = Node(1)
u = @lift un($n)
b = @lift bn($n)
c = @lift cn($n)

ax = Axis(fig[1, 1], title="Zonal velocity")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="Tracer concentration")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, c, colorrange=(0, 0.5), colormap=:thermal)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

title_str = @lift "Parameterized baroclinic adjustment at t = " * prettytime(times[$n])
ax_t = fig[0, :] = Label(fig, title_str)

display(fig)

record(fig, "zonally_averaged_baroclinic_adj.mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end

=#

#####
##### Load truth data as observations
#####

data_path = experiment_name * ".jld2"

observations = OneDimensionalTimeSeries(data_path, field_names=(:b, :c), normalize=ZScore)

#####
##### Set up ensemble model
#####

slice_ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=ensemble_size)
@show ensemble_grid = RegularRectilinearGrid(size=slice_ensemble_size, y = (-Ly/2, Ly/2), z = (-Lz, 0), topology = (Flat, Bounded, Bounded), halo=(3, 3))

closure_ensemble = [deepcopy(closures) for i = 1:ensemble_size] 
coriolis_ensemble = [BetaPlane(latitude=-45) for i = 1:ensemble_size]

ensemble_model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                             grid = ensemble_grid,
                                             tracers = (:b, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis_ensemble,
                                             closure = closure_ensemble,
                                             momentum_advection = WENO5(),
                                             tracer_advection = WENO5(),
                                             free_surface = ExplicitFreeSurface())

wall_clock = [time_ns()]
    
function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.8e, %6.8e, %6.8e) m/s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w))

    wall_clock[1] = time_ns()
    
    return nothing
end

ensemble_simulation = Simulation(ensemble_model; Δt=Δt, stop_time=stop_time, progress=print_progress, iteration_interval=1000)
pop!(ensemble_simulation.diagnostics, :nan_checker)

#####
##### Build free parameters
#####

# priors = (
#     κ_skew = lognormal_with_mean_std(900, 200),
#     κ_symmetric = lognormal_with_mean_std(1100, 200),
# )

priors = (
    κ_skew = ConstrainedNormal(0.0, 1.0, 400.0, 1200.0),
    κ_symmetric = ConstrainedNormal(0.0, 1.0, 800.0, 1800.0)
)

free_parameters = FreeParameters(priors)

###
### Visualize the prior densities
###
using CairoMakie
using OceanTurbulenceParameterEstimation.EnsembleKalmanInversions: convert_prior, inverse_parameter_transform

samples(prior) = [inverse_parameter_transform(prior, x) for x in rand(convert_prior(prior), 10000000)]
samples_κ_skew = samples(priors.κ_skew)
samples_κ_symmetric = samples(priors.κ_symmetric)

f = Figure()
axtop = Axis(f[1, 1])
densities = []
push!(densities, CairoMakie.density!(axtop, samples_κ_skew))
push!(densities, CairoMakie.density!(axtop, samples_κ_symmetric))
leg = Legend(f[1, 2], densities, ["κ_skew", "κ_symmetric"], position = :lb)
# CairoMakie.xlims!(0,2e-5)
save("visualize_prior_kappa_skew.png", f)
display(f)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# forward_map(calibration, [θ★ for _ in 1:ensemble_size])
x = forward_map(calibration, [θ★ for _ in 1:ensemble_size])
y = observation_map(calibration)

# Assert that G(θ*) ≈ y
@show x[:, 1:1] == y

#=
using Plots, LinearAlgebra
p = Plots.plot(collect(1:length(x)), [x...], label="forward_map")
Plots.plot!(collect(1:length(y)), [y...], label="observation_map")
# savefig(p, "obs_vs_pred.png")
display(p)
=#

iterations = 10
eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)
params = iterate!(eki; iterations = iterations)

@show params

###
### Summary plots
###

using LinearAlgebra

θ̅(iteration) = [eki.iteration_summaries[iteration].ensemble_mean...]
varθ(iteration) = eki.iteration_summaries[iteration].ensemble_variance

weight_distances = [norm(θ̅(iter) - θ★) for iter in 1:iterations]
output_distances = [norm(forward_map(calibration, [θ̅(iter) for _ in 1:ensemble_size])[:, 1] - y) for iter in 1:iterations]
ensemble_variances = [varθ(iter) for iter in 1:iterations]

x = 1:iterations
f = CairoMakie.Figure()
CairoMakie.lines(f[1, 1], x, weight_distances, color = :red,
            axis = (title = "Parameter distance", xlabel = "Iteration, n", ylabel="|θ̅ₙ - θ⋆|"))
CairoMakie.lines(f[1, 2], x, output_distances, color = :blue,
            axis = (title = "Output distance", xlabel = "Iteration, n", ylabel="|G(θ̅ₙ) - y|"))
ax3 = Axis(f[2, 1:2], title = "Parameter convergence", xlabel = "Iteration, n", ylabel="Ensemble variance")
for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances,i)
    CairoMakie.lines!(ax3, 1:iterations, ev / ev[1], label=String(pname))
end
CairoMakie.axislegend(ax3, position = :rt)
CairoMakie.save("summary_makie.png", f)

###
### Plot ensemble density with time
###

f = CairoMakie.Figure()
axtop = CairoMakie.Axis(f[1, 1])
axmain = CairoMakie.Axis(f[2, 1], xlabel = "κ_skew", ylabel = "κ_symmetric")
axright = CairoMakie.Axis(f[2, 2])
s = eki.iteration_summaries
scatters = []
for i in [1, 2, 5, 10]
    ensemble = transpose(s[i].parameters)
    push!(scatters, CairoMakie.scatter!(axmain, ensemble))
    CairoMakie.density!(axtop, ensemble[:, 1])
    CairoMakie.density!(axright, ensemble[:, 2], direction = :y)
end
vlines!(axmain, [κ_skew], color=:red)
vlines!(axtop, [κ_skew], color=:red)
hlines!(axmain, [κ_symmetric], color=:red)
hlines!(axright, [κ_symmetric], color=:red)
colsize!(f.layout, 1, Fixed(300))
colsize!(f.layout, 2, Fixed(200))
rowsize!(f.layout, 1, Fixed(200))
rowsize!(f.layout, 2, Fixed(300))
leg = Legend(f[1, 2], scatters, ["Initial ensemble", "Iteration 1", "Iteration 5", "Iteration 10"], position = :lb)
hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)
save("distributions_makie.png", f)

tupified_params = NamedTuple{calibration.free_parameters.names}(Tuple(params))

OceanTurbulenceParameterEstimation.InverseProblems.run_simulation_with_params!(calibration, [tupified_params for _ in 1:ensemble_size])

model_time_series = OceanTurbulenceParameterEstimation.InverseProblems.transpose_model_output(calibration.time_series_collector, calibration.observations)

model_time_series.field_time_serieses.b
