#####
##### OneDimensionalTimeSeries
#####

"""
    struct OneDimensionalTimeSeries{FT, F, G, C, D, UU, VV, TT, SS}

A time series of horizontally-averaged observational or LES data
gridded as Oceananigans fields.
"""
struct OneDimensionalTimeSeries{P, F, G, T, I, R, N} <: AbstractTimeSeries
    file_path::P
    fields::F
    grid::G
    times::T
    time_range::R
    name::N
end

const OneDimensionalTimeSeriesBatch = Vector{<:OneDimensionalTimeSeries}

function OneDimensionalTimeSeries(file_path, field_names=(:u, :v, :b, :e), time_range=nothing, Nz=64)

    file = jldopen(file_path, "r")

    αg = file["buoyancy/equation_of_state/α"] * file["buoyancy/gravitational_acceleration"]

    name = get_parameter(datapath, "parameters", "name", "")

    iters = get_iterations(datapath)

    simulation_grid = grid_type(datapath)
    u = [ new_field(XFaceField, simulation_grid, get_data("u", datapath, iter)) for iter in iters ] 
    v = [ new_field(YFaceField, simulation_grid, get_data("v", datapath, iter)) for iter in iters ] 
    b = [ new_field(CenterField, simulation_grid, get_data("T", datapath, iter) .* αg) for iter in iters ] 
    e = [ new_field(CenterField, simulation_grid, get_data("e", datapath, iter)) for iter in iters ] 

    times = get_times(datapath)

    model_grid = OneDimensionalEnsembleGrid(datapath; size=(1,1,Nz))

    # deserialize as much as possible into metadata
    meta_data = Dict(group => file[group] for group in filter(n -> n != "timeseries", keys(file)))

    fields = NamedTuple(field_name => eval(field_name) for field_name in field_names)

    time_range = isnothing(time_range) ? range(times[1], stop=times[end]) : time_range

    observation = OneDimensionalTimeSeries(file_path, fields, simulation_grid, times, time_range, name)

    return OneDimensionalTimeSeries(observation, model_grid)
end

"""
    OneDimensionalTimeSeries(observation::OneDimensionalTimeSeries, grid)

Returns `observation::OneDimensionalTimeSeries` interpolated to `grid`.
"""
function OneDimensionalTimeSeries(data::OneDimensionalTimeSeries, grid::AbstractGrid)

    U = [ XFaceField(grid) for _ in data.time_range ]
    V = [ YFaceField(grid) for _ in data.time_range ]
    B = [ CenterField(grid) for _ in data.time_range ]
    E = [ CenterField(grid) for _ in data.time_range ]

    for i in data.time_range
        set!(U[i], data.fields.u[i])
        set!(V[i], data.fields.v[i])
        set!(B[i], data.fields.b[i])
        set!(E[i], data.fields.e[i])
    end

    return OneDimensionalTimeSeries(data.file_path,
                      data.fields,
                      grid,
                      data.times[data.time_range],
                      data.time_range,
                      data.name)
end 