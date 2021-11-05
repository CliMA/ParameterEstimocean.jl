# # Convective adjustment perfect model calibration -- Ensemble Kalman Inversion
#
# This example showcases a "perfect model calibration" of the convective adjustement problem. We use
# output for buoyancy (``b``) to calibrate the convective adjustment closure and recover the 
# background and the convective diffusivities used to create some synthetic data.
#
# The calibration is done here using Ensemble Kalman Inversion. For more information about the 
# algorithm refer to [EnsembleKalmanProcesses.jl documentation](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/ensemble_kalman_inversion/).

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
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using Distributions
using LinearAlgebra

# ## Set up the problem and generate observations

# Define the "true" values for convective and background diffusivities. These are the
# parameter values that we use to generate the data. Then, we'll see if EKI calibration
# can recover the diffusivity values.

convective_κz = 1.0     # [m² s⁻¹] convective diffusivity
background_κz = 1e-4    # [m² s⁻¹] background diffusivity

convective_νz = 0.9     # [m² s⁻¹] convective viscosity
background_νz = 1e-5    # [m² s⁻¹] background viscosity
nothing #hide

# We gather the "true" parameters in a vector ``θ_*``:

θ★ = [convective_κz, background_κz]

# The experiment name and where the synthetic observations will be saved.
experiment_name = "convective_adjustment_eki_example"
data_path = experiment_name * ".jld2"

# The domain, number of grid points, and other parameters.

Nz = 32                 # grid points in the vertical
Lz = 64                 # [m] depth
Qᵇ =  1e-8              # [m² s⁻³] buoyancy flux
Qᵘ = -1e-5              # [m² s⁻²] momentum flux
Δt = 10.0               # [s] time step
f₀ = 1e-4               # [s⁻¹] Coriolis frequency
N² = 1e-6               # [s⁻²] buoyancy frequency
stop_time = 10hour      # length of run
save_interval = 1hour   # save observations every so often

generate_observations = true
nothing #hide

# ## Generate synthetic observations

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

# ## Load truth data as observations

@show observations = OneDimensionalTimeSeries(data_path, field_names=(:b,), normalize=ZScore)

observations = [observations, observations]
nothing #hide

# ## Calibration with Ensemble Kalman Inversions

# ### Ensemble model

# First we set up an ensemble model,
ensemble_size = 50

column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(ensemble_size, length(observations)), Hz=1)

@show ensemble_grid = RegularRectilinearGrid(size=column_ensemble_size,
                                             topology = (Flat, Flat, Bounded),
                                             z = (-Lz, 0))

closure_ensemble = [ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz, convective_νz, background_νz) 
                    for i = 1:ensemble_grid.Nx, j = 1:ensemble_grid.Ny]

coriolis_ensemble = [FPlane(f=f₀) for i = 1:ensemble_grid.Nx, j = 1:ensemble_grid.Ny]

ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))
ensemble_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

@show ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                                   tracers = :b,
                                                   buoyancy = BuoyancyTracer(),
                                                   boundary_conditions = (; u=ensemble_u_bcs, b=ensemble_b_bcs),
                                                   coriolis = coriolis_ensemble,
                                                   closure = closure_ensemble)

set!(ensemble_model, b = (x, y, z) -> N² * z)

# and then we create an ensemble simulation.

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

pop!(ensemble_simulation.diagnostics, :nan_checker)

ensemble_simulation

# We removed the `nan_checker` checker since we would ideally want to be able to proceed with the
# Ensemble Kalman Inversion (EKI) iteration step even if one of the models of the ensemble ends up
# blowing up.

# ### Free parameters
#
# We construct some prior distributions for our free parameters. We found that it often helps to
# constrain the prior distributions so that neither very high nor very low values for diffusivities
# can be drawn out of the distribution.

priors = (
    convective_κz = lognormal_with_mean_std(0.3, 0.5),
    background_κz = lognormal_with_mean_std(2.5e-4, 0.25e-4),
)

free_parameters = FreeParameters(priors)

# To visualize the prior distributions we randomly sample out values from then and plot the p.d.f.

using CairoMakie
using OceanTurbulenceParameterEstimation.EnsembleKalmanInversions: convert_prior, inverse_parameter_transform

samples(prior) = [inverse_parameter_transform(prior, x) for x in rand(convert_prior(prior), 50000000)]

samples_convective_κz = samples(priors.convective_κz)
samples_background_κz = samples(priors.background_κz)

f = Figure()
axtop = Axis(f[1, 1],
             xlabel = "convective_κz [m² s⁻¹]",
             ylabel = "p.d.f.")
axbottom = Axis(f[2, 1],
                xlabel = "background_κz [m² s⁻¹]",
                ylabel = "p.d.f.")
densities = []
push!(densities, density!(axtop, samples_convective_κz))
push!(densities, density!(axbottom, samples_background_κz))
xlims!(axtop, 0, 20)
save("visualize_prior_diffusivities_convective_adjustment_uki.svg", f); nothing #hide 

# ![](visualize_prior_diffusivities_convective_adjustment_eki.svg)

# ### The inverse problem

# We can construct the inverse problem ``y = G(θ) + η``. Here, ``y`` are the `observations` and `G` is the
# `ensemble_model`.
calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# ### Assert that ``G(θ_*) ≈ y``
#
# As a sanity check we apply the `forward_map` on the calibration after we initialize all ensemble
# members with the true parameter values. We then confirm that the output of the `forward_map` matches
# the observations to machine precision.

x = forward_map(calibration, θ★)
y = observation_map(calibration)

# The `forward_map` output `x` is a two-dimensional matrix whose first dimension is the size of the state space
# and whose second dimension is the `ensemble_size`. In the case above, all columns of `x` are identical.

@show mean(x, dims=2) == y

# Next, we construct an `EnsembleKalmanInversion` (EKI) object,

noise_variance = observation_map_variance_across_time(calibration)[1, :, 1] .+ 1e-5

eki = EnsembleKalmanInversion(calibration; noise_covariance=Matrix(Diagonal(noise_variance)));

# and perform few iterations to see if we can converge to the true parameter values.

iterations = 10

params = iterate!(eki; iterations = iterations)

@show params

# Last, we visualize the outputs of EKI calibration.

output_size = length(x)
indices = 1:output_size
Nx, Ny, Nz = size(calibration.time_series_collector.grid)
Nt = (Int(stop_time / save_interval) + 1)
n = Ny * Nz

v = observation_map_variance_across_time(calibration)[1, :, :]

f = Figure()
ax = Axis(f[1, 1])
for t = 0:Nt-1
    range = (t*n + 1):((t + 1)*n)
    plot!(x[range], range, color=:red, legend=false)
    plot!(0.1 .* v[range], range, color = :green)
end
save("output_with_variance_convective_adjustment_eki.svg", f); nothing #hide 

# ![](output_with_variance_convective_adjustment_eki.svg)


θ̅(iteration) = [eki.iteration_summaries[iteration].ensemble_mean...]
varθ(iteration) = eki.iteration_summaries[iteration].ensemble_variance

weight_distances = [norm(θ̅(iter) - θ★) for iter in 1:iterations]
output_distances = [norm(forward_map(calibration, θ̅(iter))[:, 1] - y) for iter in 1:iterations]
ensemble_variances = [varθ(iter) for iter in 1:iterations]

f = Figure()
lines(f[1, 1], 1:iterations, weight_distances, color = :red,
      axis = (title = "Parameter distance",
              xlabel = "Iteration",
              ylabel = "|θ̅ₙ - θ⋆|",
              yscale = log10))
lines(f[1, 2], 1:iterations, output_distances, color = :blue,
      axis = (title = "Output distance",
              xlabel = "Iteration",
              ylabel="|G(θ̅ₙ) - y|",
              yscale = log10))
ax3 = Axis(f[2, 1:2],
           title = "Parameter convergence",
           xlabel = "Iteration",
           ylabel = "Ensemble variance",
           yscale = log10)

for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances, i)
    lines!(ax3, 1:iterations, ev / ev[1], label=String(pname))
end
axislegend(ax3, position = :rt)
save("summary_convective_adjustment_eki.svg", f); nothing #hide 

# ![](summary_convective_adjustment_eki.svg)

# And also we plot the the distributions of the various model ensembles for few EKI iterations to see
# if and how well they converge to the true diffusivity values.

f = Figure()
axtop = Axis(f[1, 1])
axmain = Axis(f[2, 1],
              xlabel = "convective_κz [m² s⁻¹]",
              ylabel = "background_κz [m² s⁻¹]")
axright = Axis(f[2, 2])
scatters = []
for i in [1, 2, 3, 11]
    ensemble = transpose(eki.iteration_summaries[i].parameters)
    push!(scatters, scatter!(axmain, ensemble))
    density!(axtop, ensemble[:, 1])
    density!(axright, ensemble[:, 2], direction = :y)
end
vlines!(axmain, [convective_κz], color=:red)
vlines!(axtop, [convective_κz], color=:red)
hlines!(axmain, [background_κz], color=:red)
hlines!(axright, [background_κz], color=:red)
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
save("distributions_convective_adjustment_eki.svg", f); nothing #hide 

# ![](distributions_convective_adjustment_eki.svg)
