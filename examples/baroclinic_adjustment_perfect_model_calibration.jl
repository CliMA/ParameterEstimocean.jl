# # Baroclinic adjustment perfect model calibration
#
# This example showcases a "perfect model calibration" of the two-dimensional baroclinic adjustement
# problem # (depth-latitude) with eddies parametrized by Gent-McWilliams--Redi isoneutral diffusion.

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, Distributions, CairoMakie, OceanTurbulenceParameterEstimation"
# ```

# First we load few things

using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: FluxTapering
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Distributions
using Printf
using LinearAlgebra: norm

# ## Set up the problem and generate observations

# Define the  "true" skew and symmetrid diffusivity coefficients. These are the parameter values that we
# use to generate the data. Then, we'll see if the EKI calibration can recover these values.

κ_skew = 1000.0       # [m² s⁻¹] skew diffusivity
κ_symmetric = 900.0   # [m² s⁻¹] symmetric diffusivity
nothing #hide

# We gather the "true" parameters in a vector `\theta_★`:

θ★ = [κ_skew, κ_symmetric]

# The experiment name and where the synthetic observations will be saved.
experiment_name = "baroclinic_adjustment"
data_path = experiment_name * ".jld2"

# The domain, number of grid points, and other parameters.
architecture = CPU()      # CPU or GPU?

Ly = 1000kilometers       # north-south extent [m]
Lz = 1kilometers          # depth [m]

Ny = 64                   # grid points in north-south direction
Nz = 16                   # grid points in the vertical

Δt = 10minute             # time-step

stop_time = 1days         # length of run
save_interval = 0.25days  # save observation every so often

generate_observations = true

# The isopycnal skew-symmetric diffusivity closure.
gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_skew,
                                                                κ_symmetric = κ_symmetric,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
nothing #hide

# ## Generate synthetic observations

if generate_observations || !(isfile(data_path))
    grid = RegularRectilinearGrid(topology = (Flat, Bounded, Bounded), 
                                  size = (Ny, Nz), 
                                  y = (-Ly/2, Ly/2),
                                  z = (-Lz, 0),
                                  halo = (3, 3))
    
    model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                        grid = grid,
                                        tracers = (:b, :c),
                                        buoyancy = BuoyancyTracer(),
                                        coriolis = BetaPlane(latitude=-45),
                                        closure = gent_mcwilliams_diffusivity,
                                        free_surface = ImplicitFreeSurface())
    
    @info "Built $model."

    ##### Initial conditions of an unstable buoyancy front

    """
    Linear ramp from 0 to 1 between -Δy/2 and +Δy/2.

    For example:

    y < y₀           => ramp = 0
    y₀ < y < y₀ + Δy => ramp = y / Δy
    y > y₀ + Δy      => ramp = 1
    """
    ramp(y, Δy) = min(max(0, y/Δy + 1/2), 1)

    N² = 4e-6             # [s⁻²] buoyancy frequency / stratification
    M² = 8e-8             # [s⁻²] horizontal buoyancy gradient

    Δy = 50kilometers     # horizontal extent of the font

    Δc_y = 2Δy            # horizontal extent of initial tracer concentration
    Δc_z = 50             # [m] vertical extent of initial tracer concentration

    Δb = Δy * M²          # inital buoyancy jump

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
    
    simulation = Simulation(model, Δt=Δt, stop_time=stop_time, progress=print_progress, iteration_interval=48)
    
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = TimeInterval(save_interval),
                                                          prefix = experiment_name,
                                                          array_type = Array{Float64},
                                                          field_slicer = nothing,
                                                          force = true)

    pop!(simulation.diagnostics, :nan_checker)      # remove the `nan_checker`
    
    run!(simulation)
end

# ## Load truth data as observations

observations = OneDimensionalTimeSeries(data_path, field_names=(:b, :c), normalize=ZScore)

# ## Calibration with Ensemble Kalman Inversions

# ### Ensemble model

# First we set up an ensemble model,
ensemble_size = 10

slice_ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=ensemble_size)
@show ensemble_grid = RegularRectilinearGrid(size=slice_ensemble_size,
                                             topology = (Flat, Bounded, Bounded),
                                             y = (-Ly/2, Ly/2),
                                             z = (-Lz, 0),
                                             halo=(3, 3))

closure_ensemble = [deepcopy(gent_mcwilliams_diffusivity) for i = 1:ensemble_size] 

@show ensemble_model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                                   grid = ensemble_grid,
                                                   tracers = (:b, :c),
                                                   buoyancy = BuoyancyTracer(),
                                                   coriolis = BetaPlane(latitude=-45),
                                                   closure = closure_ensemble,
                                                   free_surface = ImplicitFreeSurface())

# and an ensemble simulation. We remove the `nan_checker` checker since we would ideally want to
# be able to proceed with the EKI iteration step even if one of the models of the ensemble ends
# up blowing up.

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

pop!(ensemble_simulation.diagnostics, :nan_checker)

ensemble_simulation

# ### Free parameters
#
# We construct some prior distributions for our free parameters. We found that it often helps to
# constrain the prior distributions so that neither very high nor very low values for diffusivities
# can be drawn out of the distribution.

priors = (
    κ_skew = ConstrainedNormal(0.0, 1.0, 400.0, 1300.0),
    κ_symmetric = ConstrainedNormal(0.0, 1.0, 700.0, 1700.0)
)

free_parameters = FreeParameters(priors)

# # We may visualize the prior distributions by randomly sampling out of them.

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
save("visualize_prior_kappa_skew.svg", f); nothing #hide 

# ![](visualize_prior_kappa_skew.svg)


# ### The inverse problem

# We can construct the inverse problem ``y = G(θ) + η``. Here, ``y`` are the `observations` and `G` is the
# `ensemble_model`.
calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# ### Assert that ``G(θ_*) ≈ y``
#
# As a sanity check we apply the `forward_map` on the calibration after we initialize all ensemble
# members with the true parameter values. We then confirm that the output of the `forward_map` matches
# the observations to machine precision.

x = forward_map(calibration, [θ★ for _ in 1:ensemble_size])
y = observation_map(calibration)

# The `forward_map` output `x` is a two-dimensional matrix whose first dimension is the size of the state space
# (here, ``2 N_y N_z``) and whose second dimension is the `ensemble_size`. In the case above, all columns of `x`
# are identical.

mean(x, dims=2) ≈ y


# Next, we construct an `EnsembleKalmanInversion` (EKI) object,

eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)

# and perform few iterations to see if we can converge to the true parameter values.

iterations = 5
params = iterate!(eki; iterations = iterations)

@show params

# Last, we visualize few metrics regarding how the EKI calibration went about.

θ̅(iteration) = [eki.iteration_summaries[iteration].ensemble_mean...]
varθ(iteration) = eki.iteration_summaries[iteration].ensemble_variance

weight_distances = [norm(θ̅(iter) - θ★) for iter in 1:iterations]
output_distances = [norm(forward_map(calibration, [θ̅(iter) for _ in 1:ensemble_size])[:, 1] - y) for iter in 1:iterations]
ensemble_variances = [varθ(iter) for iter in 1:iterations]

f = Figure()
lines(f[1, 1], 1:iterations, weight_distances, color = :red, linewidth = 2,
      axis = (title = "Parameter distance",
              xlabel = "Iteration",
              ylabel="|θ̅ₙ - θ⋆|",
              yscale = log10))
lines(f[1, 2], 1:iterations, output_distances, color = :blue, linewidth = 2,
      axis = (title = "Output distance",
              xlabel = "Iteration",
              ylabel="|G(θ̅ₙ) - y|",
              yscale = log10))
ax3 = Axis(f[2, 1:2], title = "Parameter convergence",
           xlabel = "Iteration",
           ylabel = "Ensemble variance",
           yscale = log10)

for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances,i)
    lines!(ax3, 1:iterations, ev / ev[1], label = String(pname), linewidth = 2)
end

axislegend(ax3, position = :rt)
save("summary.svg", f); nothing #hide 

# ![](summary.svg)

# And also we plot the the distributions of the various model ensembles for few EKI iterations to see
# if and how well they converge to the true diffusivity values.

f = Figure()
axtop = Axis(f[1, 1])
axmain = Axis(f[2, 1],
              xlabel = "κ_skew",
              ylabel = "κ_symmetric")
axright = Axis(f[2, 2])
summaries = eki.iteration_summaries
scatters = []
for iteration in [1, 2, 3, 6]
    ensemble = transpose(summaries[iteration].parameters)
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
leg = Legend(f[1, 2], scatters,
             ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration 5"],
             position = :lb)
hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)
xlims!(axmain, 350, 1350)
xlims!(axtop, 350, 1350)
ylims!(axmain, 650, 1750)
ylims!(axright, 650, 1750)
xlims!(axright, 0, 0.06)
ylims!(axtop, 0, 0.06)
save("distributions.svg", f); nothing #hide 

# ![](distributions.svg)
