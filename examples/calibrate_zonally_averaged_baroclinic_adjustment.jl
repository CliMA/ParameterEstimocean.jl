# Calibration of Gent-McWilliams to a baroclinic adjustment problem

using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Oceananigans.TurbulenceClosures: FluxTapering
using LinearAlgebra, CairoMakie, DataDeps, JLD2
# using ElectronDisplay

architecture = CPU()

# filedir = @__DIR__
# filename = "baroclinic_adjustment_double_Lx_zonal_average.jld2"
# filepath = joinpath(filedir, filename)
# Base.download("https://www.dropbox.com/s/f8zsb33vwwwmjjm/$filename", filepath)

# filepath = "/Users/navid/Research/mesoscale-parametrization-OSM2022/baroclinic_adjustment-double_Lx/short_save_often_run/baroclinic_adjustment_double_Lx_zonal_average.jld2"
filepath = "baroclinic_adjustment_double_Lx_zonal_average.jld2"

file = jldopen(filepath)
coriolis = file["serialized/coriolis"]

# number of grid points
Nx, Ny, Nz = file["grid/Nx"], file["grid/Ny"], file["grid/Nz"]

# Domain
const Lx, Ly, Lz = file["grid/Lx"], file["grid/Ly"], file["grid/Lz"]

close(file)


field_names = (:b, :c, :u)

using OceanTurbulenceParameterEstimation.Transformations: Transformation

transformation = (b = ZScore(),
                  c = ZScore(),
                  u = ZScore())

transformation = ZScore()

space_transformation = SpaceIndices(x=:, y=2:2:Ny-1, z=2:2:Nz-1)

transformation = (b = Transformation(space = space_transformation, normalization=ZScore()),
                  c = Transformation(space = space_transformation, normalization=ZScore()),
                  u = Transformation(space = space_transformation, normalization=ZScore()))

# transformation = ZScore()

times = [0, 1hours]

observations = SyntheticObservations(filepath; transformation, times, field_names)

#####
##### Simulation
#####

Nensemble = 5
slice_ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=Nensemble)

ensemble_grid = RectilinearGrid(architecture,
                                size = slice_ensemble_size,
                                topology = (Flat, Bounded, Bounded),
                                y = (-Ly/2, Ly/2),
                                z = (-Lz, 0),
                                halo=(3, 3))

@show ensemble_grid

file = jldopen(filepath)
closures = file["serialized/closure"]
close(file)

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(slope_limiter = FluxTapering(1e-2))

closure_ensemble = ([deepcopy(gent_mcwilliams_diffusivity) for _ = 1:Nensemble], closures[1], closures[2])

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = (:b, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis,
                                             closure = closure_ensemble,
                                             free_surface = ImplicitFreeSurface())

Δt = 5.0
simulation = Simulation(ensemble_model; Δt, stop_time=times[end])

priors = (
     κ_skew = ScaledLogitNormal(bounds = (300, 5000)),
     κ_symmetric = ScaledLogitNormal(bounds = (300, 5000))
 )

free_parameters = FreeParameters(priors)

calibration = InverseProblem(observations, simulation, free_parameters)

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 1e-5,
                              resampler = Resampler(acceptable_failure_fraction=0.3))

iterate!(eki; iterations = 5)
