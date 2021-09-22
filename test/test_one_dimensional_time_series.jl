using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

using OceanTurbulenceParameterEstimation

#####
##### Parameters
#####

Nz = 16
Lz = 128
Qᵇ = 1e-8
Qᵘ = -1e-5
Δt = 20.0
f₀ = 1e-4
N² = 1e-5

stop_time = 6Δt
save_interval = 2Δt
N_ensemble = 3

experiment_name = "convective_adjustment_test"
data_path = experiment_name * ".jld2"

# "True" parameters to be estimated by calibration
convective_κz = 1.0
convective_νz = 0.9
background_κz = 1e-4
background_νz = 1e-5

@testset "OneDimensionalTimeSeries" begin
    #####
    ##### Generate synthetic observations
    #####

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

    #####
    ##### Load truth data as observations
    #####

    data_path = experiment_name * ".jld2"

    b_observations = OneDimensionalTimeSeries(data_path, field_names=:b)
    ub_observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :b))
    uvb_observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :v, :b))

    @test keys(b_observations.fields) == tuple(:b)
    @test keys(ub_observations.fields) == tuple(:u, :b)
    @test keys(uvb_observations.fields) == tuple(:u, :v, :b)
end
