# # Calibrate zonally-averaged channel from 3D eddying channel output

# ## Install dependencies

# ```julia
# using Pkg
# pkg"add OceanLearning, Oceananigans, Distributions, CairoMakie"
# ```

using OceanLearning
using OceanLearning.Observations: FieldTimeSeriesCollector

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

file = jldopen(filepath)

# number of grid points
Nx, Ny, Nz = file["grid/Nx"], file["grid/Ny"], file["grid/Nz"]

# Domain
const Lx, Ly, Lz = file["grid/Lx"], file["grid/Ly"], file["grid/Lz"]

close(file)

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

end_time = 60day

times = [0, end_time]

u_timeseries = get_field_timeseries(filepath, "u", times)
v_timeseries = get_field_timeseries(filepath, "v", times)
η_timeseries = get_field_timeseries(filepath, "η", times)
b_timeseries = get_field_timeseries(filepath, "b", times)
c_timeseries = get_field_timeseries(filepath, "c", times)

#=
field_names = (:b, :c, :u, :v, :η)

transformation = (b = ZScore(),
                  c = ZScore(),
                  u = ZScore(),
                  v = RescaledZScore(1e-2),
                  η = RescaledZScore(1e-2))

field_time_serieses = (b = b_timeseries, c = c_timeseries, u = u_timeseries, v = v_timeseries, η = η_timeseries)
=#

# let's try fewer fields
field_names = (:b, :u)

transformation = (b = ZScore(),
                  u = ZScore())

field_time_serieses = (b = b_timeseries, u = u_timeseries) 

observations = SyntheticObservations(; transformation, times, field_names, field_time_serieses)

#####
##### Simulation
#####

file = jldopen(filepath)
coriolis = file["serialized/coriolis"]
close(file)

Nensemble = 5
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
    κ_skew = ScaledLogitNormal(bounds = (300, 1200)),
    κ_symmetric = ScaledLogitNormal(bounds = (300, 1200))
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

# Last, we visualize few metrics regarding how the EKI calibration went about.

θ̅(iteration) = [eki.iteration_summaries[iteration].ensemble_mean...]
varθ(iteration) = eki.iteration_summaries[iteration].ensemble_var

weight_distances = [norm(θ̅(iter)) for iter in 1:eki.iteration]
output_distances = [norm(forward_map(calibration, θ̅(iter))[:, 1] - y) for iter in 1:eki.iteration]
ensemble_variances = [varθ(iter) for iter in 1:eki.iteration]

f = Figure()
lines(f[1, 1], 1:eki.iteration, weight_distances, color = :red, linewidth = 2,
      axis = (title = "Parameter norm",
              xlabel = "Iteration",
              ylabel="|θ̅ₙ|",
              yscale = log10))
lines(f[1, 2], 1:eki.iteration, output_distances, color = :blue, linewidth = 2,
      axis = (title = "Output distance",
              xlabel = "Iteration",
              ylabel="|G(θ̅ₙ) - y|",
              yscale = log10))
ax3 = Axis(f[2, 1:2], title = "Parameter convergence",
           xlabel = "Iteration",
           ylabel = "Ensemble variance",
           yscale = log10)

for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances, i)
    lines!(ax3, 1:eki.iteration, ev / ev[1], label = String(pname), linewidth = 2)
end

axislegend(ax3, position = :rt)
save("summary_channel.svg", f); nothing #hide 

# ![](summary_channel.svg)

# And also we plot the the distributions of the various model ensembles for few EKI iterations to see
# if and how well they converge to the true diffusivity values.

f = Figure()

axtop = Axis(f[1, 1])

axmain = Axis(f[2, 1],
              xlabel = "κ_skew [m² s⁻¹]",
              ylabel = "κ_symmetric [m² s⁻¹]")

axright = Axis(f[2, 2])
scatters = []
labels = String[]

for iteration in [0, 1, 2, 5]
    ## Make parameter matrix
    parameters = eki.iteration_summaries[iteration].parameters
    Nensemble = length(parameters)
    Nparameters = length(first(parameters))
    parameter_ensemble_matrix = [parameters[i][j] for i=1:Nensemble, j=1:Nparameters]

    label = iteration == 0 ? "Initial ensemble" : "Iteration $iteration"
    push!(labels, label)
    push!(scatters, scatter!(axmain, parameter_ensemble_matrix))
    density!(axtop, parameter_ensemble_matrix[:, 1])
    density!(axright, parameter_ensemble_matrix[:, 2], direction = :y)
end

vlines!(axmain, [κ_skew], color = :red)
vlines!(axtop, [κ_skew], color = :red)

hlines!(axmain, [κ_symmetric], color = :red)
hlines!(axright, [κ_symmetric], color = :red)

colsize!(f.layout, 1, Fixed(300))
colsize!(f.layout, 2, Fixed(200))

rowsize!(f.layout, 1, Fixed(200))
rowsize!(f.layout, 2, Fixed(300))

Legend(f[1, 2], scatters, labels, position = :lb)

hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)

xlims!(axmain, 350, 1350)
xlims!(axtop, 350, 1350)
ylims!(axmain, 650, 1750)
ylims!(axright, 650, 1750)
xlims!(axright, 0, 0.025)
ylims!(axtop, 0, 0.025)

save("distributions_channel.svg", f); nothing #hide 

# ![](distributions_channel.svg)
