# # Convective adjustment perfect model calibration -- Unscented Kalman Inversion
#
# This example showcases a "perfect model calibration" of the convective adjustement problem. We use
# output for buoyancy (``b``) to calibrate the convective adjustment closure and recover the 
# background and the convective diffusivities used to create some synthetic data.
#
# The calibration is done here using Unscented Kalman Inversion. For more information about the 
# algorithm refer to [EnsembleKalmanProcesses.jl documentation](https://clima.github.io/EnsembleKalmanProcesses.jl/stable/unscented_kalman_inversion/).

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, EnsembleKalmanProcesses, CairoMakie"
# ```

# First we load few things

using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using Distributions
using EnsembleKalmanProcesses.ParameterDistributionStorage
using LinearAlgebra

# ## Set up the problem and generate observations

# Define the "true" values for convective and background diffusivities. These are the
# parameter values that we use to generate the data. Then, we'll see if UKI calibration
# can recover the diffusivity values.

convective_κz = 1.0     # [m² s⁻¹] convective diffusivity
background_κz = 1e-4    # [m² s⁻¹] background diffusivity

convective_νz = 0.9     # [m² s⁻¹] convective viscosity
background_νz = 1e-5    # [m² s⁻¹] background viscosity
nothing #hide

# We gather the "true" parameters in a vector ``θ_*``:

θ★ = [convective_κz, background_κz]

# The experiment name and where the synthetic observations will be saved.
experiment_name = "convective_adjustment_uki_example"
data_path = experiment_name * ".jld2"

# The domain, number of grid points, and other parameters.

Nz = 32                 # grid points in the vertical
Lz = 128                # [m] depth
Qᵇ =  1e-8              # [m² s⁻³] buoyancy flux
Qᵘ = -1e-5              # [m² s⁻²] momentum flux
Δt = 20.0               # [s] time step
f₀ = 1e-4               # [s⁻¹] Coriolis frequency
N² = 1e-5               # [s⁻²] buoyancy frequency
stop_time = 10hour      # length of run
save_interval = 1hour   # save observations every so often

generate_observations = true
nothing #hide

# ## Generate synthetic observations

if generate_observations || !(isfile(data_path))
    grid = RectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
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

observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :b), normalize=ZScore)

# ## Calibration with Unscented Kalman Inversions (UKI)

# ### Ensemble model

# First we set up an ensemble model. UKI uses ``2 N_θ + 1`` particles, where ``N_θ`` is the
# total number of the free parameters to be calibrated.

Nθ = length(θ★)

ensemble_size = 2Nθ + 1

column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(ensemble_size, 1), Hz=1)

@show ensemble_grid = RectilinearGrid(size=column_ensemble_size,
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

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

pop!(ensemble_simulation.diagnostics, :nan_checker)

ensemble_simulation

# We removed the `nan_checker` checker since we would ideally want to be able to proceed with the
# Unscented Kalman Inversion (UKI) iteration step even if one of the models of the ensemble ends up
# blowing up.

# ### Free parameters
#
# We construct some prior distributions for our free parameters. We found that it often helps to
# constrain the prior distributions so that neither very high nor very low values for diffusivities
# can be drawn out of the distribution.

priors = (
    convective_κz = ConstrainedNormal(0.0, 1.0, 0.0, 4*convective_κz),
    background_κz = ConstrainedNormal(0.0, 1.0, 0.0, 4*background_κz)
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
save("visualize_prior_diffusivities_convective_adjustment_uki.svg", f); nothing #hide 

# ![](visualize_prior_diffusivities_convective_adjustment_uki.svg)

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
# and whose second dimension is the `ensemble_size`. In the case above, all columns of `x` are identical.

@show mean(x, dims=2) ≈ y

# Next, we construct an `UscentedKalmanInversion` (UKI) object,

prior_mean = fill(0.0, Nθ) 
prior_cov = Matrix(Diagonal(fill(1.0, Nθ)))
α_reg = 1.0   # regularization parameter 
update_freq = 1
noise_covariance = 0.05^2  # error is about 5%

uki = UnscentedKalmanInversion(calibration, prior_mean, prior_cov;
                               noise_covariance = noise_covariance, α_reg = α_reg, update_freq = update_freq)

# and perform few iterations to see if we can converge to the true parameter values.

iterations = 10

iterate!(uki; iterations = iterations)

# Last, we visualize the outputs of UKI calibration.

θ_mean, θθ_cov, θθ_std_arr, error =  UnscentedKalmanInversionPostprocess(uki)

N_iter = size(θ_mean, 2)

f = Figure(resolution = (800, 800))
ax1 = Axis(f[1, 1],
           xlabel = "iterations",
           xticks = 1:N_iter,
           ylabel = "convective_κz [m² s⁻¹]")
ax2 = Axis(f[2, 1],
           xlabel = "iterations",
           xticks = 1:N_iter,
           ylabel = "background_κz [m² s⁻¹]")
ax3 = Axis(f[3, 1],
           xlabel = "iterations",
           ylabel = "error",
           xticks = 1:N_iter)

lines!(ax1, 1:N_iter, θ_mean[1, :])
band!(ax1, 1:N_iter, θ_mean[1, :] .+ θθ_std_arr[1, :], θ_mean[1, :] .- θθ_std_arr[1, :])
hlines!(ax1, [convective_κz], color=:red)

lines!(ax2, 1:N_iter, θ_mean[2, :])
band!(ax2, 1:N_iter, θ_mean[2, :] .+ θθ_std_arr[2, :], θ_mean[2, :] .- θθ_std_arr[2, :])
hlines!(ax2, [background_κz], color=:red)

plot!(ax3, 2:N_iter, error)

xlims!(ax1, 0.5, N_iter+0.5)
xlims!(ax2, 0.5, N_iter+0.5)
xlims!(ax3, 0.5, N_iter+0.5)

save("uki_results.svg", f); nothing #hide 

# ![](uki_results.svg)
