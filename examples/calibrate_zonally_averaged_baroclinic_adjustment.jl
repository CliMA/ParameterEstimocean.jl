# Calibration of Gent-McWilliams to a baroclinic adjustment problem

using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Oceananigans.TurbulenceClosures: FluxTapering
using LinearAlgebra, CairoMakie, DataDeps, JLD2
using ElectronDisplay

architecture = CPU()

# filedir = @__DIR__
# filename = "baroclinic_adjustment_double_Lx_zonal_average.jld2"
# filepath = joinpath(filedir, filename)
# Base.download("https://www.dropbox.com/s/f8zsb33vwwwmjjm/$filename", filepath)

filepath = "/Users/navid/Research/mesoscale-parametrization-OSM2022/baroclinic_adjustment-double_Lx/baroclinic_adjustment_double_Lx_zonal_average.jld2"

field_names = (:b, :c, :u)

normalization = (b = ZScore(),
                 c = ZScore(),
                 u = ZScore())

times = [0, 12hours]

observations = SyntheticObservations(filepath; normalization, times, field_names)

#####
##### Simulation
#####

file = jldopen(filepath)
coriolis = file["serialized/coriolis"]
close(file)

# Domain
Ly = observations.grid.Ly
Lz = observations.grid.Lz

# number of grid points
Ny = observations.grid.Ny
Nz = observations.grid.Nz

Nensemble = 6
slice_ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=Nensemble)

ensemble_grid = RectilinearGrid(architecture,
                                size = slice_ensemble_size,
                                topology = (Flat, Bounded, Bounded),
                                y = (-Ly/2, Ly/2),
                                z = (-Lz, 0),
                                halo=(3, 3))

@show ensemble_grid

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(slope_limiter = FluxTapering(1e-2))
closure_ensemble = [deepcopy(gent_mcwilliams_diffusivity) for k = 1:Nensemble]

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = (:b, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis,
                                             closure = closure_ensemble,
                                             free_surface = ImplicitFreeSurface())

Δt = 10.0
simulation = Simulation(ensemble_model; Δt, stop_time=times[end])

priors = (
     κ_skew = ScaledLogitNormal(bounds = (300, 500)),
     κ_symmetric = ScaledLogitNormal(bounds = (300, 500))
 )

free_parameters = FreeParameters(priors)
calibration = InverseProblem(observations, simulation, free_parameters)
eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-1)
