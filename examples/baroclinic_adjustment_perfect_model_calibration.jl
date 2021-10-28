pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Distributions
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
Ny = 128
Nz = 32

architecture = CPU()

stop_time = 40days
Δt = 5minutes
stop_time = 40days
save_interval = 2hour
experiment_name = "baroclinic_adjustment"
data_path = experiment_name * ".jld2"
ensemble_size = 10
generate_observations = true

# "True" parameters to be estimated by calibration
κ_skew = 1000       # [m² s⁻¹] skew diffusivity
κ_symmetric = 1000  # [m² s⁻¹] symmetric diffusivity


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

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = 1000,
                                                                κ_symmetric = (b=0, c=1000),
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)
                                        
closures = (diffusive_closure, convective_adjustment, gent_mcwilliams_diffusivity)

#####
##### Generate synthetic observations
#####

if generate_observations || !(isfile(data_path))
    grid = RegularRectilinearGrid(topology = (Periodic, Bounded, Bounded), 
                                  size = (Nx, Ny, Nz), 
                                  x = (0, Lx),
                                  y = (-Ly/2, Ly/2),
                                  z = (-Lz, 0),
                                  halo = (3, 3, 3))

    coriolis = BetaPlane(latitude=-45)

    
    model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                        grid = grid,
                                        coriolis = coriolis,
                                        buoyancy = BuoyancyTracer(),
                                        closure = closures,
                                        tracers = (:b, :c),
                                        momentum_advection = WENO5(),
                                        tracer_advection = WENO5(),
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
##### Visualize
#####

fig = Figure(resolution = (1400, 700))

filepath = "zonally_averaged_baroclinic_adj_fields.jld2"

ut = FieldTimeSeries(filepath, "u")
bt = FieldTimeSeries(filepath, "b")
ct = FieldTimeSeries(filepath, "c")
rt = FieldTimeSeries(filepath, "Rb")

# Build coordinates, rescaling the vertical coordinate
x, y, z = nodes((Center, Center, Center), grid)

zscale = 1
z = z .* zscale

#####
##### Plot buoyancy...
#####

times = bt.times
Nt = length(times)

un(n) = interior(ut[n])[1, :, :]
bn(n) = interior(bt[n])[1, :, :]
cn(n) = interior(ct[n])[1, :, :]
rn(n) = interior(rt[n])[1, :, :]

@show min_c = 0
@show max_c = 1
@show max_u = maximum(abs, un(Nt))
min_u = - max_u

@show max_r = maximum(abs, rn(Nt))
@show min_r = - max_r

n = Node(1)
u = @lift un($n)
b = @lift bn($n)
c = @lift cn($n)
r = @lift rn($n)

ax = Axis(fig[1, 1], title="Zonal velocity")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, u, colorrange=(min_u, max_u), colormap=:balance)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="Tracer concentration")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, c, colorrange=(0, 0.5), colormap=:thermal)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[2, 2], hm)

ax = Axis(fig[3, 1], title="R(b)")
hm = heatmap!(ax, y * 1e-3, z * 1e-3, r, colorrange=(min_r, max_r), colormap=:balance)
contour!(ax, y * 1e-3, z * 1e-3, b, levels = 25, color=:black, linewidth=2)
cb = Colorbar(fig[3, 2], hm)

title_str = @lift "Parameterized baroclinic adjustment at t = " * prettytime(times[$n])
ax_t = fig[0, :] = Label(fig, title_str)

display(fig)

record(fig, "zonally_averaged_baroclinic_adj.mp4", 1:Nt, framerate=8) do i
    @info "Plotting frame $i of $Nt"
    n[] = i
end







#####
##### Load truth data as observations
#####

data_path = experiment_name * ".jld2"

observations = OneDimensionalTimeSeries(data_path, field_names=(:b, :c), normalize=ZScore)

#####
##### Set up ensemble model
#####

slice_ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=ensemble_size, halo=(1, 1))
ensemble_grid = RegularRectilinearGrid(size=slice_ensemble_size, y = (0, Ly), z = (-Lz, 0), topology = (Flat, Bounded, Bounded))

closure_ensemble = [deepcopy(closures) for i = 1:ensemble_size] 
coriolis_ensemble = [BetaPlane(-45) for i = 1:ensemble_size]

ensemble_model = HydrostaticFreeSurfaceModel(architecture = architecture,
                                             grid = ensemble_grid,
                                             tracers = (:b, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis_ensemble,
                                             closure = closure_ensemble,
                                             momentum_advection = WENO5(),
                                             tracer_advection = WENO5(),
                                             free_surface = ImplicitFreeSurface())

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)
pop!(ensemble_simulation.diagnostics, :nan_checker)

#####
##### Build free parameters
#####

priors = (
    κ_skew = lognormal_with_mean_std(900, 200),
    κ_symmetric = lognormal_with_mean_std(1100, 200),
)

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# forward_map(calibration, [θ★ for _ in 1:ensemble_size])
x = forward_map(calibration, [θ★ for _ in 1:ensemble_size])[1:1, :]
y = observation_map(calibration)

#=
using Plots, LinearAlgebra
# p = plot(collect(1:length(x)), [x...], label="forward_map")
# plot!(collect(1:length(y)), [y...], label="observation_map")
# savefig(p, "obs_vs_pred.png")
# display(p)

# Assert that G(θ*) ≈ y
@show forward_map(calibration, [θ★ for _ in 1:ensemble_size]) == observation_map(calibration)


iterations = 10
eki = EnsembleKalmanInversion(calibration; noise_covariance=1e-2)
params, mean_vars, mean_us = iterate!(eki; iterations = iterations)

@show params
y = eki.mapped_observations
a = [norm(forward_map(calibration, [mean_us[i] for _ in 1:ensemble_size])[:,1] - y) for i in 1:iterations]
plot(collect(1:iterations), a)
=#