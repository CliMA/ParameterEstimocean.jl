pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Oceananigans
using Plots, LinearAlgebra, Distributions
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using OceanTurbulenceParameterEstimation

include("lesbrary_paths.jl")
include("parameters.jl")

#####
##### Simulation parameters
#####

Nz = 32
Lz = 64
Qᵇ = 1e-8
Qᵘ = -1e-5
Δt = 10.0
f₀ = 1e-4
N² = 1e-6
stop_time = 10hour
save_interval = 1hour
experiment_name = "catke_perfect_model_observation"
data_path = experiment_name * ".jld2"
ensemble_size = 50
generate_observations = true

#####
##### Generate synthetic observations
#####

parameter_set = CATKEParametersRiDependent

println(CATKEParametersRiDependent.defaults)

true_closure = closure_with_parameters(ConvectiveAdjustmentVerticalDiffusivity(Float64;), parameter_set)

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

observations = SixDaySuite("/Users/adelinehillier/.julia/dev")

#####
##### Set up ensemble model
#####

closure = closure_with_parameters(CATKEVerticalDiffusivity(Float64; warning=false), all_defaults)

ensemble_model = OneDimensionalEnsembleModel(observations::OneDimensionalTimeSeriesBatch; 
                       architecture = CPU(), 
                       ensemble_size = 50, 
                       closure = closure
                      )

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

pop!(ensemble_simulation.diagnostics, :nan_checker)

#####
##### Build free parameters
#####

free_parameters = CATKEParametersRiDependent
priors = (pname = ConstrainedNormal(0.0, 1.0, bounds(pname)...) for pname in free_parameters)

θ★ = [convective_κz, background_κz, convective_νz, background_νz]

free_parameters = FreeParameters(pr)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

forward_map(calibration, [[1.0, 0.9, 1e-4, 1e-5],
                          [1.1, 0.9, 1e-4, 1e-5],
                          [1.0, 1.0, 1e-4, 1e-5],
                          [1.0, 0.9, 2e-4, 1e-5],
                          [1.0, 0.9, 1e-4, 2e-5]]
)