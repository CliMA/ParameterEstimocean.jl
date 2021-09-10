module OceanTurbulenceParameterEstimation

export OneDimensionalTimeSeries, InverseProblem

using Oceananigans
using Oceananigans: short_show
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields
using JLD2

include("free_parameters.jl")

abstract type AbstractObservation end

"""
    OneDimensionalTimeSeries{F, G, T, P, M} <: AbstractObservation

A time series of horizontally-averaged observational or LES data
gridded as Oceananigans fields.
"""
struct OneDimensionalTimeSeries{F, G, T, P, M} <: AbstractObservation
       fields :: F
         grid :: G
        times :: T
         path :: P
     metadata :: M
end

tupleit(t) = try
    Tuple(t)
catch
    tuple(t)
end

const not_metadata_names = ("serialized", "timeseries")

read_group(group::JLD2.Group) = NamedTuple(Symbol(subgroup) => read_group(group[subgroup]) for subgroup in keys(group))
read_group(group) = group

function OneDimensionalTimeSeries(path; field_names)
    field_names = tupleit(field_names)
    fields = NamedTuple(name => FieldTimeSeries(path, string(name)) for name in field_names)
    grid = fields[1].grid
    times = fields[1].times

    # validate_data(fields, grid, times) # might be a good idea to validate the data...
    file = jldopen(path)
    metadata = NamedTuple(Symbol(group) => read_group(file[group]) for group in filter(n -> n ∉ not_metadata_names, keys(file)))
    close(file)

    return OneDimensionalTimeSeries(fields, grid, times, path, metadata)
end

Base.show(io::IO, ts::OneDimensionalTimeSeries) =
    print(io, "OneDimensionalTimeSeries with fields $(propertynames(ts.fields))", '\n',
              "├── times: $(ts.times)", '\n',    
              "├── grid: $(short_show(ts.grid))", '\n',
              "├── path: \"$(ts.path)\"", '\n',
              "└── metadata: $(keys(ts.metadata))")

#
# InverseProblem
#

#=
using Oceananigans: AbstractModel

get_model_closure(model::AbstractModel) = get_model_closure(model.closure)
get_model_closure(closure) = closure
get_model_closure(closure::AbstractArray) = CUDA.@allowscalar closure[1, 1]

defaults(model::AbstractModel, RelevantParameters) = DefaultFreeParameters(get_model_closure(model), RelevantParameters)

function InverseProblem(observations::OneDimensionalTimeSeriesBatch, simulation::Simulation, parameters::Parameters{UnionAll}; transformation = )

    simulation = Simulation(model; Δt = Δt, stop_time = 0.0)
    pop!(simulation.diagnostics, :nan_checker)

    # Set model to custom defaults
    set!(model, custom_defaults(model, parameters.RelevantParameters))

    default_parameters = custom_defaults(model, parameters.ParametersToOptimize)

    return InverseProblem(observations, simulation, parameters)
end
=#

# const OneDimensionalTimeSeriesBatch = Vector{<:OneDimensionalTimeSeries}

#=

"""
    OneDimensionalTimeSeries(observation::OneDimensionalTimeSeries, grid)

Returns `observation::OneDimensionalTimeSeries` interpolated to `grid`.
"""
function OneDimensionalTimeSeries(data::OneDimensionalTimeSeries, grid::AbstractGrid)

    U = [ XFaceField(grid) for t in data.times ]
    V = [ YFaceField(grid) for t in data.times ]
    B = [ CenterField(grid) for t in data.times ]
    E = [ CenterField(grid) for t in data.times ]

    for i = 1:length(data.t)
        set!(U[i], data.fields.u[i])
        set!(V[i], data.fields.v[i])
        set!(B[i], data.fields.b[i])
        set!(E[i], data.fields.e[i])
    end

    return OneDimensionalTimeSeries(data.path,
                      data.fields,
                      grid,
                      data.times,
                      data.time_range,
                      data.name)
end
=#


#=
export

    # Grids
    OneDimensionalEnsembleGrid,

    # Observations
    OneDimensionalTimeSeries, OneDimensionalTimeSeriesBatch,

    # Models
    get_model_field, OneDimensionalEnsembleModel, ensemble_size, batch_size,
    set!, initialize_forward_run!,
    DefaultFreeParameters, get_free_parameters, FreeParameters, @free_parameters,

    # LossFunctions
    LossFunction,
    TimeSeriesAnalysis,
    TimeAverage,
    model_time_series,
    ForwardMap,
    ValueProfileAnalysis,
    GradientProfileAnalysis,

    # ParameterEstimation
    InverseProblem,
    Parameters
    
include("Grids/Grids.jl")
include("Observations/Observations.jl")
include("Models/Models.jl")
include("LossFunctions/LossFunctions.jl")
include("ParameterEstimation/ParameterEstimation.jl")

using .Grids
using .Observations
using .Models
using .LossFunctions
using .ParameterEstimation
=#

end # module
