#####
##### ParameterizedModel
#####

mutable struct ParameterizedModel{M<:AbstractModel, T}
    model :: M
       Δt :: T
end

run_until!(pm::ParameterizedModel, time) = run_until!(pm.model, pm.Δt, time)

Base.getproperty(m::ParameterizedModel, ::Val{:Δt}) = getfield(m, :Δt)
Base.getproperty(m::ParameterizedModel, ::Val{:model}) = getfield(m, :model)
Base.getproperty(m::ParameterizedModel, p::Symbol) = getproperty(m, Val(p))
# Base.getproperty(m::ParameterizedModel, ::Val{p}) where p = getproperty(m.model, p)

function Base.getproperty(m::ParameterizedModel, ::Val{p}) where p

    p ∈ propertynames(m.model.tracers) && return m.model.tracers[p]

    p ∈ propertynames(m.model.velocities) && return m.model.velocities[p]

    return getproperty(m.model, p)

end

#####
##### TruthData
#####

"""
    struct TruthData{FT, F, G, C, D, UU, VV, TT, SS}

A time series of horizontally-averaged observational or LES data
gridded as Oceananigans fields.
"""
struct TruthData{F, G, C, D, UU, VV, BΘ, EE, TT, NN}
   boundary_conditions :: F
                  grid :: G
             constants :: C
         diffusivities :: D
                     u :: UU
                     v :: VV
                     b :: BΘ
                     e :: EE
                     t :: TT
                  name :: NN
end

"""
    TruthData(datapath)

Construct TruthData from a time-series of Oceananigans LES data saved at `datapath`.
"""
function TruthData(datapath; grid_type=ZGrid,
                             Nz=32)

    # For now, we assume salinity-less LES data.
    file = jldopen(datapath, "r")

    constants = Dict()

    constants[:α] = file["buoyancy/equation_of_state/α"]
    constants[:g] = file["buoyancy/gravitational_acceleration"]
    constants[:αg] = constants[:α] * constants[:g]
    # constants[:β] = 0.0 #file["buoyancy/equation_of_state/β"]

    constants[:f] = 0.0
    try
        constants[:f] = file["coriolis/f"]
    catch end

    close(file)

    # Surface fluxes
    Qᵘ = get_parameter(datapath, "parameters", "boundary_condition_u_top", 0.0)
    Qᵛ = get_parameter(datapath, "parameters", "boundary_conditions_v_top", 0.0)
    Qᶿ = get_parameter(datapath, "parameters", "boundary_condition_θ_top", 0.0)

    # Bottom gradients
    dθdz_bottom = get_parameter(datapath, "parameters", "boundary_condition_θ_bottom", 0.0)
    dudz_bottom = get_parameter(datapath, "parameters", "boundary_condition_u_bottom", 0.0)

    name = get_parameter(datapath, "parameters", "name", "")

    background_ν = get_parameter(datapath, "closure", "ν")
    background_κ = (T=get_parameter(datapath, "closure/κ", "T"),)

    iters = get_iterations(datapath)

    simulation_grid = grid_type(datapath)
    u = [  XFaceField(simulation_grid, get_data("u", datapath, iter)) for iter in iters ]
    v = [  YFaceField(simulation_grid, get_data("v", datapath, iter)) for iter in iters ]
    b = [ CenterField(simulation_grid, get_data("T", datapath, iter) .* constants[:αg]) for iter in iters ]
    e = [ CenterField(simulation_grid, get_data("e", datapath, iter)) for iter in iters ]

    for (i, iter) in enumerate(iters)
        u² = XFaceField(simulation_grid, get_data("uu", datapath, iter))
        v² = YFaceField(simulation_grid, get_data("vv", datapath, iter))
        w² = ZFaceField(simulation_grid, get_data("ww", datapath, iter))

        N = simulation_grid.Nz
        # e_interior = interior(e[i])
        # @. e_interior = ( interior(u²) - interior(u[i])^2 + interior(v²) - interior(v[i])^2 + interior(w²) ) / 2
        @. e[i].data[1:N] = ( u²[1:N] - u[i][1:N]^2 + v²[1:N] - v[i][1:N]^2
                                + 1/2 * (w²[1:N] + w²[2:N+1]) ) / 2
    end

    t = get_times(datapath)

    td = TruthData((Qᶿ=Qᶿ, Qᵘ=Qᵘ, Qᵛ=Qᵛ, Qᵉ=0.0, dθdz_bottom=dθdz_bottom, dudz_bottom=dudz_bottom),
                      simulation_grid, constants, (ν=background_ν, κ=background_κ),
                      u, v, b, e, t, name)

    model_grid = grid_type(datapath; size=Nz)

    # Return TruthData with grid and variables coarse_grained to model resolution
    td_coarse = TruthData(td, model_grid)

    return td_coarse

end

"""
    TruthData(data::TruthData, grid)

Returns `data::TruthData` interpolated to `grid`.
"""
function TruthData(td::TruthData, grid::AbstractGrid)

    U = [ XFaceField(grid) for t in td.t ]
    V = [ YFaceField(grid) for t in td.t ]
    B = [ CenterField(grid) for t in td.t ]
    E = [ CenterField(grid) for t in td.t ]

    for i = 1:length(td.t)
        set!(U[i], td.u[i])
        set!(V[i], td.v[i])
        set!(B[i], td.b[i])
        set!(E[i], td.e[i])
    end

    return TruthData(td.boundary_conditions,
                      grid,
                      td.constants,
                      td.diffusivities,
                      U,
                      V,
                      B,
                      E,
                      td.t,
                      td.name)
end

"""
    TruthData(data::TruthData, grid)

Construct TruthData from a time-series of Oceananigans LES data saved at `datapath`.
and interpolate the data to `grid`.
"""
function TruthData(datapath, grid::AbstractGrid)
    td = TruthData(datapath)
    return TruthData(td, grid)
end

length(td::TruthData) = length(td.t)

function time_step!(model, Δt, Nt)
    for step = 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

time(model) = model.clock.time
iteration(model) = model.clock.iteration

"""
    run_until!(model, Δt, tfinal)
Run `model` until `tfinal` with time-step `Δt`.
"""
function run_until!(model, Δt, tfinal)
    Nt = floor(Int, (tfinal - time(model))/Δt)
    time_step!(model, Δt, Nt)

    last_Δt = tfinal - time(model)
    last_Δt == 0 || time_step!(model, last_Δt)

    return nothing
end

function initialize_forward_run!(model, data, params, index)
    set!(model, params)
    set!(model, data, index)
    model.clock.iteration = 0
    return nothing
end

# function initialize_and_run_until!(model, data, parameters, initial, target)
#     initialize_forward_run!(model, data, parameters, initial)
#     run_until!(model.model, model.Δt, data.t[target])
#     return nothing
# end