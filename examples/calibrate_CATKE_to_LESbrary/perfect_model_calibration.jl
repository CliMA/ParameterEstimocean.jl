pushfirst!(LOAD_PATH, joinpath(@__DIR__, "../.."))

using Oceananigans
using Plots, LinearAlgebra, Distributions, JLD2
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using OceanTurbulenceParameterEstimation

include("lesbrary_paths.jl")
include("parameters.jl")

#####
##### Simulation parameters
#####

Nz = 32
Lz = 64
Qᵇ = 6e-8
Qᵘ = -7e-4
Δt = 10.0
f₀ = 1e-4
N² = 1e-5
stop_time = 3hour
save_interval = 1hour
experiment_name = "catke_perfect_model_observation"
data_path = experiment_name * ".jld2"
ensemble_size = 50
generate_observations = true

#####
##### Generate synthetic observations
#####

parameter_set = CATKEParametersRiDependent
closure = closure_with_parameter_set(CATKEVerticalDiffusivity(Float64;), parameter_set)

true_parameters = (Cᵟu = 0.5, CᴷRiʷ = 1.0, Cᵂu★ = 2.0, CᵂwΔ = 1.0, Cᴷeʳ = 5.0, Cᵟc = 0.5, Cᴰ = 2.0, Cᴷc⁻ = 0.5, Cᴷe⁻ = 0.2, Cᴷcʳ = 3.0, Cᴸᵇ = 1.0, CᴷRiᶜ = 1.0, Cᴷuʳ = 4.0, Cᴷu⁻ = 1.2, Cᵟe = 0.5)
true_closure = closure_with_parameters(closure, true_parameters)

if generate_observations || !(isfile(data_path))

    grid = RegularRectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        tracers = (:b, :e),
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (; u=u_bcs, b=b_bcs),
                                        coriolis = FPlane(f=f₀),
                                        closure = true_closure)
                                        
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

data_path2 = "catke_perfect_model_observation2.jld2"
if generate_observations || !(isfile(data_path2))

    grid = RegularRectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ*0.5), bottom = GradientBoundaryCondition(N²))

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        tracers = (:b, :e),
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (; u=u_bcs, b=b_bcs),
                                        coriolis = FPlane(f=f₀),
                                        closure = true_closure)
                                        
    set!(model, b = (x, y, z) -> N² * z)
    
    simulation = Simulation(model; Δt, stop_time)
    
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = TimeInterval(save_interval),
                                                          prefix = "catke_perfect_model_observation2",
                                                          array_type = Array{Float64},
                                                          field_slicer = nothing,
                                                          force = true)
    
    run!(simulation)
end

#####
##### Load truth data as observations
#####

data_path = experiment_name * ".jld2"

# observations = SixDaySuite("/Users/adelinehillier/.julia/dev")

observation1 = OneDimensionalTimeSeries(data_path, field_names=(:b, :e, :u, :v), normalize=ZScore)
observation2 = OneDimensionalTimeSeries(data_path2, field_names=(:b, :e, :u, :v), normalize=ZScore)
observations = [observation1, observation2]

Nx = ensemble_size
Ny = length(observations)
column_ensemble_size = ColumnEnsembleSize(Nz=Nz, ensemble=(Nx, Ny), Hz=1)

#####
##### Set up ensemble model
#####

ensemble_grid = RegularRectilinearGrid(size=column_ensemble_size, z = (-Lz, 0), topology = (Flat, Flat, Bounded))
closure_ensemble = [closure for i = 1:Nx, j = 1:Ny]

function get_parameter(observation, parameter_path)
    file = jldopen(observation.path)
    # parameter = getproperty(file, parameter_path)
    parameter = file[parameter_path]
    close(file)
    return parameter
end

get_Qᵇ(observation) = get_parameter(observation, "timeseries/b/serialized/boundary_conditions").top.condition
get_Qᵘ(observation) = get_parameter(observation, "timeseries/u/serialized/boundary_conditions").top.condition
get_f₀(observation) = get_parameter(observation, "coriolis/f")

Qᵇ_ensemble = [get_Qᵇ(observations[j]) for i = 1:Nx, j = 1:Ny]
Qᵘ_ensemble = [get_Qᵘ(observations[j]) for i = 1:Nx, j = 1:Ny]

ensemble_b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ_ensemble), bottom = GradientBoundaryCondition(N²))
ensemble_u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
coriolis_ensemble = [FPlane(f=get_f₀(observations[j])) for i = 1:Nx, j = 1:Ny]

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = (:b, :e),
                                             buoyancy = BuoyancyTracer(),
                                             boundary_conditions = (; u=ensemble_u_bcs, b=ensemble_b_bcs),
                                             coriolis = coriolis_ensemble,
                                             closure = closure_ensemble)

set!(ensemble_model, b = (x, y, z) -> N² * z)

ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

pop!(ensemble_simulation.diagnostics, :nan_checker)

# #####
# ##### Set up ensemble model
# #####

# ensemble_model = OneDimensionalEnsembleModel(observations; 
#                        architecture = CPU(), 
#                        ensemble_size = 50, 
#                        closure = closure
#                       )

# ensemble_simulation = Simulation(ensemble_model; Δt, stop_time)

# pop!(ensemble_simulation.diagnostics, :nan_checker)

#####
##### Build free parameters
#####

free_parameter_names = keys(parameter_set.defaults)
free_parameter_means = [values(parameter_set.defaults)...]
priors = NamedTuple(pname => ConstrainedNormal(0.0, 1.0, bounds(pname)...) for pname in free_parameter_names)

θ★ = [values(true_parameters)...]

free_parameters = FreeParameters(priors)

#####
##### Build the Inverse Problem
#####

calibration = InverseProblem(observations, ensemble_simulation, free_parameters)

a = forward_map(calibration, free_parameter_means)
x = forward_map(calibration, θ★)
y = observation_map(calibration)

