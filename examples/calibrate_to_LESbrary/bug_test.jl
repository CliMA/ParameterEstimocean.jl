using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using OceanTurbulenceParameterEstimation.Observations: FieldTimeSeriesCollector, IdentityNormalization, SyntheticObservations
using OceanTurbulenceParameterEstimation.InverseProblems: ConcatenatedOutputMap, transform_time_series

Ex, Ey = (2, 2)
sz = ColumnEnsembleSize(Nz = 8, ensemble = (Ex, Ey))
halo = ColumnEnsembleSize(Nz = sz.Nz)

grid = RectilinearGrid(size = sz, halo = halo, z = (-128, 0), topology = (Flat, Flat, Bounded))

closure = [ConvectiveAdjustmentVerticalDiffusivity() for i = 1:Ex, j = 1:Ey]

Qᵇ = [+1e-8 for i = 1:Ex, j = 1:Ey]
Qᵘ = [-1e-4 for i = 1:Ex, j = 1:Ey]
Qᵛ = [0.0 for i = 1:Ex, j = 1:Ey]

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(grid = grid,
    tracers = (:b, :e),
    buoyancy = BuoyancyTracer(),
    coriolis = FPlane(f = 1e-4),
    boundary_conditions = (b = b_bcs, u = u_bcs, v = v_bcs),
    closure = closure)

N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

Δt = 10.0
stop_iteration = 3
times = [Δt * (n - 1) for n = 1:stop_iteration]

field_names = (:u,)

simulation = Simulation(model; Δt = Δt, stop_iteration = stop_iteration)

simulation_fields = fields(simulation.model)

fields_to_collect = NamedTuple(name => simulation_fields[name] for name in field_names)

time_series_collector = FieldTimeSeriesCollector(fields_to_collect, times)

function initialize!(simulation, time_series_collector)

    model_fields = fields(simulation.model)
    for name in keys(model_fields)
        model_field = model_fields[name]
        set!(model_field, 0.0)
    end

    set!(simulation.model, b = bᵢ)

    simulation.model.clock.time = 0.0
    simulation.model.clock.iteration = 0

    for time_series in time_series_collector.field_time_serieses
        time_series.data .= 0
    end

    simulation.callbacks[:data_collector] = Callback(time_series_collector, SpecifiedTimes(times...))

    simulation.stop_time = times[end]

    return nothing
end

function forward_map(simulation, time_series_collector)

    initialize!(simulation, time_series_collector)
    run!(simulation)

    map = ConcatenatedOutputMap()

    normalization = Dict(name => IdentityNormalization() for name in field_names)

    transposed_output = SyntheticObservations(time_series_collector.field_time_serieses,
        time_series_collector.grid,
        time_series_collector.times,
        nothing,
        nothing,
        normalization)

    output = transform_time_series(map, transposed_output)

    return output
end

x = forward_map(simulation, time_series_collector)[:, 1:1]

y = forward_map(simulation, time_series_collector)[:, 1:1]

@show x == y