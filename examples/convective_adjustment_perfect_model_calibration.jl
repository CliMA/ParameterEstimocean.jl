pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using OceanTurbulenceParameterEstimation

#####
##### Parameters
#####

Nz = 32
Lz = 8
Qᵇ = 1e-8
Δt = 20.0
f₀ = 1e-4
stop_time = 12hours
save_interval = 1hour
experiment_name = "convective_adjustment"
N_ensemble = 10

# Free parameters of the problem
convective_κz = 1.0
convective_νz = 0.9
background_κz = 1e-4
background_νz = 1e-5

#####
##### Generate "truth" data
#####

#=
grid = RegularRectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
closure = ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz, convective_νz, background_νz)
coriolis = FPlane(f=f₀)
                                      
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (; b=b_bcs),
                                    coriolis = coriolis,
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

simulation = Simulation(model; Δt, stop_time)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      schedule = TimeInterval(save_interval),
                                                      prefix = experiment_name,
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)
=#

data_path = experiment_name * ".jld2"

#####
##### Load truth data as observations
#####

observation = OneDimensionalTimeSeries(data_path, field_names=:b)

#####
##### Set up ensemble model
#####

ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(N_ensemble, 1), Hz=1)
ensemble_grid = RegularRectilinearGrid(size=ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
closure_ensemble = [ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz) for i = 1:N_ensemble, j = 1:1]
coriolis_ensemble = [FPlane(f=f₀) for i = 1:N_ensemble, j = 1:1]

ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = :b,
                                             buoyancy = BuoyancyTracer(),
                                             boundary_conditions = (; b=ensemble_b_bcs),
                                             coriolis = coriolis_ensemble,
                                             closure = closure_ensemble)

N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(ensemble_model, b = bᵢ)

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

# @free_parameters convective_adj_free_params convective_κz convective_νz background_κz background_νz

convective_adj_free_params = @free_parameters convective_κz convective_νz background_κz background_νz

struct FreeParameters
    value
    prior
end
ConvectiveAdjustmentParameters = FreeParameters(<:FreeParameter)





convective_κz_meta = ParameterMeta

parameters = ConvectiveAdjustmentParameters()

ConvectiveAdjustmentParameters isa FreeParameters{}

closure = ConvectiveAdjustmentVerticalDiffusivity()

stability_fn_parameters_priors = StabilityFnParameters(
    CᴷRiʷ = Normal(0.25, 0.05) |> logify,
    CᴷRiᶜ = Normal(0.25, 0.5) |> logify,
    Cᴷu⁻ = Uniform(0.0, 10.0),
    Cᴷuʳ = Uniform(0.0, 10.0),
    Cᴷc⁻ = Uniform(0.0, 10.0),
    Cᴷcʳ = Uniform(0.0, 10.0),
    Cᴷe⁻ = Uniform(0.0, 10.0),
    Cᴷeʳ = Uniform(0.0, 10.0)
)

free_parameters = FreeParameters(parameters = StabilityFnParameters,
                                 prior = stability_fn_parameters_priors

calibration = InverseProblem(ensemble_simulation, observations)

transformation = NothingTransformation()

simple_calibration_problem = InverseProblem(model, observation; transformation, parameters)