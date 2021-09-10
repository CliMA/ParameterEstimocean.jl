
using Printf
using Oceananigans
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

#####
##### Parameters
#####

Nz = 32
Lz = 64
Qᵇ = 1e-8
Δt = 20.0
stop_time = 24hours
save_interval = 1hour
experiment_name = "convective_adjustment"
N_ensemble = 10

# Free parameters of the problem
convective_κz = 1.0
background_κz = 1e-4

#####
##### Generate "truth" data
#####

grid = RegularRectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
closure = ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz)
                                      
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; b=b_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

simulation = Simulation(model; Δt, stop_time)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(save_interval),
                                                      prefix = experiment_name,
                                                      force = true)

run!(simulation)

data_path = experiment_name * ".jld2"

#####
##### Load truth data as observations
#####

free_convection_observation = OneDimensionalTimeSeries(data_path,
                                                       field_names=tuple(:b),
                                                       time_range=range(2hours, stop=2days, step=4hours))

#####
##### Set up ensemble model
#####

ensemble_size = ColumnEnsembleSize(Nz=Nz, )
ensemble_grid = RegularRectilinearGrid



model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; b=b_bcs),
                                    closure = closure)
 