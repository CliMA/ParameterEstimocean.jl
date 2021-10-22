pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using LinearAlgebra
using Distributions
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using OceanTurbulenceParameterEstimation
using EnsembleKalmanProcesses.ParameterDistributionStorage
#####
##### Parameters
#####

Nz = 32
Lz = 128
Qᵇ = 1e-8
Qᵘ = -1e-5
Δt = 20.0
f₀ = 1e-4
N² = 1e-5
stop_time = 10hours
save_interval = 1hour
experiment_name = "convective_adjustment"
data_path = experiment_name * ".jld2"
generate_observations = false

# "True" parameters to be estimated by calibration
convective_κz = 1.0
background_κz = 1e-4
convective_νz = 0.9
background_νz = 1e-5

θ★ = [convective_κz, background_κz]

Nθ = length(θ★)
# UKI uses 2Nθ + 1 particles
ensemble_size = 2Nθ+1
#####
##### Generate synthetic observations
#####

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

#####
##### Load truth data as observations
#####

data_path = experiment_name * ".jld2"

observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :b), normalize=ZScore)

#####
##### Set up ensemble model
#####

column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(ensemble_size, 1), Hz=1)
ensemble_grid = RegularRectilinearGrid(size=column_ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
closure_ensemble = [ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz, convective_νz, background_νz) for i = 1:ensemble_grid.Nx, j = 1:ensemble_grid.Ny]
coriolis_ensemble = [FPlane(f=f₀) for i = 1:ensemble_grid.Nx, j = 1:ensemble_grid.Ny]

ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))
ensemble_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = :b,
                                             buoyancy = BuoyancyTracer(),
                                             boundary_conditions = (; u=ensemble_u_bcs, b=ensemble_b_bcs),
                                             coriolis = coriolis_ensemble,
                                             closure = closure_ensemble)

set!(ensemble_model, b = (x, y, z) -> N² * z)

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

pop!(ensemble_simulation.diagnostics, :nan_checker)
#####
##### Build free parameters
#####


priors = (
    convective_κz = ConstrainedNormal(0.0, 1.0, 0.0, 4*convective_κz),
    background_κz = ConstrainedNormal(0.0, 1.0, 0.0, 4*background_κz),
    # convective_νz = ConstrainedNormal(0.0, 1.0, 0.0, 2*convective_νz),
    # background_νz = ConstrainedNormal(0.0, 1.0, 0.0, 2*background_νz)
)


free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

# forward_map(calibration, [θ★ for _ in 1:ensemble_size])
x = forward_map(calibration, [θ★ for _ in 1:ensemble_size])[1:1, :]
y = observation_map(calibration)

using Plots, LinearAlgebra

iterations = 10

prior_mean = fill(0.0, Nθ) 
prior_cov = Matrix(Diagonal(fill(1.0, Nθ)))
α_reg = 1.0   # regularization parameter 
update_freq = 1
# error is about 5%
noise_covariance = 0.05^2
eki = UnscentedKalmanInversion(calibration, prior_mean, prior_cov; noise_covariance=noise_covariance, α_reg = α_reg, update_freq=update_freq)

iterate!(eki; iterations = iterations)

θ_mean, θθ_cov, θθ_std_arr, err =  UnscentedKalmanInversionPostprocess(eki)

# Parameter plot
N_iter = size(θ_mean, 2)
ites = Array(LinRange(1, N_iter, N_iter))
p1 = plot(ites, grid = false, θ_mean[1, :], yerror = θθ_std_arr[1, :], label = "θ1")
plot!(ites, fill(θ★[1], N_iter), linestyle = :dash, linecolor = :grey, label = nothing)

plot!(ites, grid = false, θ_mean[2, :], yerror = θθ_std_arr[2, :], label = "θ2", xaxis = "Iterations")
plot!(ites, fill(θ★[2], N_iter), linestyle = :dash, linecolor = :grey, label = nothing, yaxis=("Parameters", :log10))

# Error plot
p2 = plot(ites[1:length(err)], grid = false, err, xaxis = "Iterations", yaxis = "Error", reuse = false)
plot(p1, p2,  layout = @layout[a; b])

