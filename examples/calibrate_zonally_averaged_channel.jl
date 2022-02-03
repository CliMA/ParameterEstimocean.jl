# # Perfect CAKTE calibration with Ensemble Kalman Inversion

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, CairoMakie"
# ```

using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.Observations: FieldTimeSeriesCollector

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Oceananigans.TurbulenceClosures: FluxTapering

using LinearAlgebra, CairoMakie, DataDeps, Distributions, JLD2

using ElectronDisplay

architecture = CPU()

# download from https://www.dropbox.com/s/91altratyy1g0fc/eddying_channel_catke_zonal_average.jld2?dl=0
# and change path below accordingly
filedir = @__DIR__
filename = "eddying_channel_catke_zonal_average.jld2"
filepath = joinpath(filedir, filename)
Base.download("https://www.dropbox.com/s/91altratyy1g0fc/$filename", filepath)

b_timeseries = FieldTimeSeries(filepath, "b")

field_names = (:b, :w)
# field_names = (:b, :c, :u, :w)

normalization = (b = ZScore(),
                #  c = ZScore(),
                #  u = ZScore(),
                 w = RescaledZScore(1e-2))

times = b_timeseries.times[500:2:502]

observations = SyntheticObservations(filepath; normalization, times, field_names)

#####
##### Simulation
#####

file = jldopen(filepath)

coriolis = file["serialized/coriolis"]


# Domain
const Ly, Lz= file["grid/Ly"], file["grid/Lz"]

# number of grid points
Ny, Nz= file["grid/Ny"], file["grid/Nz"]

Nensemble = 6

slice_ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=Nensemble)

ensemble_grid = RectilinearGrid(architecture,
                                size = slice_ensemble_size,
                                topology = (Flat, Bounded, Bounded),
                                y = (0, Ly),
                                z = (-Lz, 0),
                                halo=(3, 3))

κ_skew = 1000.0       # [m² s⁻¹] skew diffusivity
κ_symmetric = 900.0   # [m² s⁻¹] symmetric diffusivity

gerdes_koberle_willebrand_tapering = FluxTapering(1e-2)
gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(κ_skew = κ_skew,
                                                                κ_symmetric = κ_symmetric,
                                                                slope_limiter = gerdes_koberle_willebrand_tapering)

closures = file["serialized/closure"]

closure_ensemble = ([deepcopy(gent_mcwilliams_diffusivity) for _ = 1:Nensemble], closures[1], closures[2])

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = (:b, :e, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis,
                                             closure = closure_ensemble,
                                             free_surface = ImplicitFreeSurface())

Δt = 5minutes
simulation = Simulation(ensemble_model; Δt, stop_time=observations.times[end])

priors = (
    κ_skew = ConstrainedNormal(0.0, 1.0, 400.0, 1300.0),
    κ_symmetric = ConstrainedNormal(0.0, 1.0, 700.0, 1700.0)
)

free_parameters = FreeParameters(priors)

collected_fields = (b = simulation.model.tracers.b,
                    w = simulation.model.velocities.w)

time_series_collector = FieldTimeSeriesCollector(collected_fields, observations.times)

calibration = InverseProblem(observations, simulation, free_parameters; time_series_collector)

eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)

