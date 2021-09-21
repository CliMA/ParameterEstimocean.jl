pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Distributions
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using OceanTurbulenceParameterEstimation

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
stop_time = 12hours
save_interval = 1hour
experiment_name = "convective_adjustment"
data_path = experiment_name * ".jld2"
N_ensemble = 10
generate_observations = false

# "True" parameters to be estimated by calibration
convective_κz = 1.0
convective_νz = 0.9
background_κz = 1e-4
background_νz = 1e-5

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
                                                          field_slicer = nothing,
                                                          force = true)
    
    run!(simulation)
end

#####
##### Load truth data as observations
#####

data_path = experiment_name * ".jld2"

observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :b))

#####
##### Set up ensemble model
#####

ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(N_ensemble, 1), Hz=1)
ensemble_grid = RegularRectilinearGrid(size=ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
closure_ensemble = [ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz) for i = 1:N_ensemble, j = 1:1]
coriolis_ensemble = [FPlane(f=f₀) for i = 1:N_ensemble, j = 1:1]

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

#####
##### Build free parameters
#####

priors = (
    convective_κz = Normal(1.2, 0.05),
    background_κz = Normal(1e-4, 1e-5),
    convective_νz = Normal(0.05, 0.01),
    background_νz = Normal(1e-4, 1e-5),
)

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

θ★ = [convective_κz, convective_νz, background_κz, background_νz]

# Assert that G(θ*) ≈ y
@assert forward_map(calibration, θ★) ≈ observation(calibration)