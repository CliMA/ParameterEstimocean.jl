# # Perfect CAKTE calibration with Ensemble Kalman Inversion

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, CairoMakie"
# ```

using OceanTurbulenceParameterEstimation

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Oceananigans.TurbulenceClosures: FluxTapering

using LinearAlgebra, CairoMakie, DataDeps, Distributions, JLD2

using ElectronDisplay

architecture = CPU()

# download from https://www.dropbox.com/s/91altratyy1g0fc/eddying_channel_catke_zonal_average.jld2?dl=0
# and change path below accordingly
filepath = "/Users/navid/Research/mesoscale-parametrization-OSM2022/eddying_channel/Ny200Nx100_Lx1000_Ly2000/eddying_channel_catke_zonal_average.jld2"

b_timeseries = FieldTimeSeries(filepath, "b")

field_names = (:b, :u, :v, :w)
# field_names = (:b, :c, :u, :v, :w)

normalization = (b = ZScore(),
                #  c = ZScore(),
                 u = ZScore(),
                 v = ZScore(),
                 w = RescaledZScore(1e-2))

times = b_timeseries.times[500:10:800]

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

closure_ensemble = [(deepcopy(gent_mcwilliams_diffusivity), closures[1], closures[2]) for _ = 1:Nensemble]

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = (:b, :e, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis,
                                             closure = closure_ensemble,
                                             free_surface = ImplicitFreeSurface(),
                                             )

#=

Δt = 5minute              # time-step
stop_time = 1year         # length of run
save_interval = 7days     # save observation every so often

simulation = Simulation(ensemble_model; Δt, stop_time)

# Ideally, we want to create a function ensemble_slice_model_simulation() and build
# the simulation via, e.g., 

closures = file["serialized/closure"]

simulation = ensemble_slice_model_simulation(observations;
                                             Nensemble = Nensemble,
                                             architecture = architecture,
                                             tracers = (:b, :e, :c),
                                             free_closures = gent_mcwilliams_diffusivity,
                                             additional_closures = closures::Tuple
                                             )

=#