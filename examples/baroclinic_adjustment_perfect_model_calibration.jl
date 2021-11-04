using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: FluxTapering
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Distributions
using Printf

using OceanTurbulenceParameterEstimation


# Here we perform a "perfect model calibration" of the two-dimensional baroclinic adjustement problem
# (depth-latitude) with eddies parametrized by Gent-McWilliams--Redi isoneutral diffusion.

# "True" parameters to be estimated by calibration
κ_skew = 1000.0       # [m² s⁻¹] skew diffusivity
κ_symmetric = 900.0   # [m² s⁻¹] symmetric diffusivity
nothign #hide

# Domain
experiment_name = "baroclinic_adjustment"

Ly = 1000kilometers # north-south extent [m]
Lz = 1kilometers    # depth [m]

Ny = 64
Nz = 16

architecture = CPU()

stop_time = 1days
save_interval = 0.25days
Δt = 12minute

ensemble_size = 10
generate_observations = true

θ★ = [κ_skew, κ_symmetric]

gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_skew,
                                                                κ_symmetric = κ_symmetric,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
                                        
coriolis = BetaPlane(latitude=-45)

# Generate synthetic observations

if generate_observations
    grid = RegularRectilinearGrid(topology = (Flat, Bounded, Bounded), 
                                  size = (Ny, Nz), 
                                  y = (-Ly/2, Ly/2),
                                  z = (-Lz, 0),
                                  halo = (3, 3))
    
    model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                        grid = grid,
                                        coriolis = coriolis,
                                        buoyancy = BuoyancyTracer(),
                                        closure = gent_mcwilliams_diffusivity,
                                        tracers = (:b, :c),
                                        free_surface = ImplicitFreeSurface())
    
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

    N² = 4e-6 # [s⁻²] buoyancy frequency / stratification
    M² = 8e-8 # [s⁻²] horizontal buoyancy gradient

    Δy = 50kilometers

    Δc_y = 2Δy
    Δc_z = 50

    Δb = Δy * M²

    bᵢ(x, y, z) = N² * z + Δb * ramp(y, Δy)
    cᵢ(x, y, z) = exp(-y^2 / 2Δc_y^2) * exp(-(z + Lz/2)^2 / (2Δc_z^2))

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
    
    simulation = Simulation(model, Δt=Δt, stop_time=stop_time, progress=print_progress, iteration_interval=40)
    
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = TimeInterval(save_interval),
                                                          prefix = experiment_name,
                                                          array_type = Array{Float64},
                                                          field_slicer = nothing,
                                                          force = true)

    pop!(simulation.diagnostics, :nan_checker)
    
    run!(simulation)
end


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

closure_ensemble = [deepcopy(gent_mcwilliams_diffusivity) for i = 1:ensemble_size] 

ensemble_model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                             grid = ensemble_grid,
                                             tracers = (:b, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis,
                                             closure = closure_ensemble,
                                             free_surface = ImplicitFreeSurface())

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)
pop!(ensemble_simulation.diagnostics, :nan_checker)

#####
##### Build free parameters
#####

priors = (
    κ_skew = ConstrainedNormal(0.0, 1.0, 400.0, 1300.0),
    κ_symmetric = ConstrainedNormal(0.0, 1.0, 700.0, 1700.0)
)

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

x = forward_map(calibration, [θ★ for _ in 1:ensemble_size])
y = observation_map(calibration)

# # Assert that G(θ*) ≈ y
@show x[:, 1:1] == y

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
push!(densities, density!(axtop, samples_κ_skew))
push!(densities, density!(axtop, samples_κ_symmetric))
leg = Legend(f[1, 2], densities, ["κ_skew", "κ_symmetric"], position = :lb)
save("assets/visualize_prior_kappa_skew.svg", f)

![](assets/visualize_prior_kappa_skew.svg)



iterations = 5
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
f = Figure()
lines(f[1, 1], x, weight_distances, color = :red, linewidth = 2,
      axis = (title = "Parameter distance", xlabel = "Iteration", ylabel="|θ̅ₙ - θ⋆|", yscale = log10))
lines(f[1, 2], x, output_distances, color = :blue, linewidth = 2,
      axis = (title = "Output distance", xlabel = "Iteration", ylabel="|G(θ̅ₙ) - y|", yscale = log10))
ax3 = Axis(f[2, 1:2], title = "Parameter convergence", xlabel = "Iteration", ylabel="Ensemble variance", yscale = log10)

for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances,i)
    lines!(ax3, 1:iterations, ev / ev[1], label=String(pname), linewidth = 2)
end

axislegend(ax3, position = :rt)
save("assets/summary.svg", f)

![](assets/summary.svg)


###
### Plot ensemble density with time
###

f = Figure()
axtop = Axis(f[1, 1])
axmain = Axis(f[2, 1], xlabel = "κ_skew", ylabel = "κ_symmetric")
axright = Axis(f[2, 2])
s = eki.iteration_summaries
scatters = []
for i in [1, 2, 3, 6]
    ensemble = transpose(s[i].parameters)
    push!(scatters, scatter!(axmain, ensemble))
    density!(axtop, ensemble[:, 1])
    density!(axright, ensemble[:, 2], direction = :y)
end
vlines!(axmain, [κ_skew], color=:red)
vlines!(axtop, [κ_skew], color=:red)
hlines!(axmain, [κ_symmetric], color=:red, alpha=0.6)
hlines!(axright, [κ_symmetric], color=:red, alpha=0.6)
colsize!(f.layout, 1, Fixed(300))
colsize!(f.layout, 2, Fixed(200))
rowsize!(f.layout, 1, Fixed(200))
rowsize!(f.layout, 2, Fixed(300))
leg = Legend(f[1, 2], scatters, ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration 5"], position = :lb)
hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)
xlims!(axmain, 400, 1400)
xlims!(axtop, 400, 1400)
ylims!(axmain, 600, 1600)
ylims!(axright, 600, 1600)
xlims!(axright, 0, 0.06)
ylims!(axtop, 0, 0.06)
save("assets/distributions_makie.svg", f)

![](assets/distributions_makie.svg)