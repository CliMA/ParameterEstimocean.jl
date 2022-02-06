# Perfect CAKTE calibration with Ensemble Kalman Inversion

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


# filepath = "/Users/navid/Research/mesoscale-parametrization-OSM2022/eddying_channel/Ny200Nx200_Lx2000_Ly2000/eddying_channel_catke_zonal_average.jld2"
# filepath = "/Users/navid/Research/mesoscale-parametrization-OSM2022/eddying_channel/eddying_channel_convadj_zonal_time_average_1year.jld2"

filedir = @__DIR__
filename = "eddying_channel_convadj_zonal_time_average_1year.jld2"
filepath = joinpath(filedir, filename)
Base.download("https://www.dropbox.com/s/8wa05l2iqnbck1z/$filename", filepath)

# number of grid points
Nx, Ny, Nz = file["grid/Nx"], file["grid/Ny"], file["grid/Nz"]

# Domain
const Lx, Ly, Lz = file["grid/Lx"], file["grid/Ly"], file["grid/Lz"]

grid = RectilinearGrid(architecture;
                       topology = (Periodic, Bounded, Bounded),
                       size = (Nx, Ny, Nz),
                       halo = (3, 3, 3),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (-Lz, 0)) # z_faces)

function get_field_timeseries(filepath, name, times)
    file = jldopen(filepath)
    
    iterations = parse.(Int, keys(file["timeseries/t"]))
    final_iteration = iterations[end]
    
    field = file["timeseries/$name/$final_iteration"]

    LX, LY, LZ = file["timeseries/$name/serialized/location"]

    close(file)

    field_timeseries = FieldTimeSeries{LX, LY, LZ}(grid, times)

    for n in 1:length(times)
        field_timeseries[n] .= field
    end

    return field_timeseries
end

end_time = 1day

times = [0, end_time]

u_timeseries = get_field_timeseries(filepath, "u", times)
v_timeseries = get_field_timeseries(filepath, "v", times)
η_timeseries = get_field_timeseries(filepath, "η", times)
b_timeseries = get_field_timeseries(filepath, "b", times)
c_timeseries = get_field_timeseries(filepath, "c", times)

#=
field_names = (:b, :c, :u, :v, :η)

normalization = (b = ZScore(),
                 c = ZScore(),
                 u = ZScore(),
                 v = RescaledZScore(1e-2),
                 η = RescaledZScore(1e-2))

field_time_serieses = (b = b_timeseries, c = c_timeseries, u = u_timeseries, v = v_timeseries, η = η_timeseries)
=#

# let's try fewer fields
field_names = (:b, :u)

normalization = (b = ZScore(),
                 u = ZScore())

field_time_serieses = (b = b_timeseries, u = u_timeseries)

observations = SyntheticObservations(;
                                     normalization,
                                     times,
                                     field_names, 
                                     field_time_serieses)

#####
##### Simulation
#####

file = jldopen(filepath)
coriolis = file["serialized/coriolis"]
close(file)

Nensemble = 1
slice_ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=Nensemble)

ensemble_grid = RectilinearGrid(architecture,
                                size = slice_ensemble_size,
                                topology = (Flat, Bounded, Bounded),
                                y = (0, Ly),
                                z = (-Lz, 0),
                                halo=(3, 3))

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(slope_limiter = FluxTapering(1e-2))

file = jldopen(filepath)
closures = file["serialized/closure"]
close(file)

closure_ensemble = ([deepcopy(gent_mcwilliams_diffusivity) for _ = 1:Nensemble], closures[1], closures[2])

# closure_ensemble = [deepcopy(gent_mcwilliams_diffusivity) for k = 1:Nensemble]

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = (:b, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis,
                                             closure = closure_ensemble,
                                             free_surface = ImplicitFreeSurface())

Δt = 5minutes
simulation = Simulation(ensemble_model; Δt, stop_time=observations.times[end])

priors = (
    κ_skew = ScaledLogitNormal(bounds = (300, 700)),
    κ_symmetric = ScaledLogitNormal(bounds = (300, 700))
)

free_parameters = FreeParameters(priors)

calibration = InverseProblem(observations, simulation, free_parameters)

eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)

#=
collected_fields = (b = simulation.model.tracers.b,
                    w = simulation.model.velocities.w)
time_series_collector = FieldTimeSeriesCollector(collected_fields, observations.times)
calibration = InverseProblem(observations, simulation, free_parameters; time_series_collector)
eki = EnsembleKalmanInversion(calibration; noise_covariance = 1e-2)
=#

iterate!(eki; iterations = 5)
