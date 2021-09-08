#####
##### OneDimensionalTimeSeries
#####

"""
    struct OneDimensionalTimeSeries{FT, F, G, C, D, UU, VV, TT, SS}

A time series of horizontally-averaged observational or LES data
gridded as Oceananigans fields.
"""
struct OneDimensionalTimeSeries{F, G, C, D, UU, VV, BΘ, EE, TT, NN, TG, RF}
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
               targets :: TG
       relevant_fields :: RF
end

"""
    OneDimensionalTimeSeries(datapath)

Construct OneDimensionalTimeSeries from a time-series of Oceananigans LES data saved at `datapath`.
"""
function OneDimensionalTimeSeries(LEScase; grid_type=ColumnEnsembleGrid,
                             Nz=32)

    datapath = LEScase.filename

    # For now, we assume salinity-less LES data.
    file = jldopen(datapath, "r")

    constants = Dict()

    constants[:α] = file["buoyancy/equation_of_state/α"]
    constants[:g] = file["buoyancy/gravitational_acceleration"]
    constants[:αg] = αg = constants[:α] * constants[:g]
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
    Qᵇ = Qᶿ * αg

    # Bottom gradients
    dudz_bottom = get_parameter(datapath, "parameters", "boundary_condition_u_bottom", 0.0)
    dθdz_bottom = get_parameter(datapath, "parameters", "boundary_condition_θ_bottom", 0.0)
    dbdz_bottom = dθdz_bottom * αg

    name = get_parameter(datapath, "parameters", "name", "")

    background_ν = get_parameter(datapath, "closure", "ν")
    background_κ = (T=get_parameter(datapath, "closure/κ", "T"),)

    iters = get_iterations(datapath)

    simulation_grid = grid_type(datapath)
    u = [ new_field(XFaceField, simulation_grid, get_data("u", datapath, iter)) for iter in iters ] 
    v = [ new_field(YFaceField, simulation_grid, get_data("v", datapath, iter)) for iter in iters ] 
    b = [ new_field(CenterField, simulation_grid, get_data("T", datapath, iter) .* constants[:αg]) for iter in iters ] 
    e = [ new_field(CenterField, simulation_grid, get_data("e", datapath, iter)) for iter in iters ] 

    t = get_times(datapath)

    last_target = isnothing(LEScase.last) ? length(t) : LEScase.last 
    targets = LEScase.first:last_target
    relevant_fields = !(LEScase.stressed) ? (:b, :e) :
                      !(LEScase.rotating) ? (:b, :u, :e) :
                                          (:b, :u, :v, :e)

    td = OneDimensionalTimeSeries((Qᶿ=Qᶿ, Qᵇ=Qᵇ, Qᵘ=Qᵘ, Qᵛ=Qᵛ, Qᵉ=0.0, 
                      dθdz_bottom=dθdz_bottom, dbdz_bottom=dbdz_bottom, dudz_bottom=dudz_bottom),
                      simulation_grid, constants, (ν=background_ν, κ=background_κ),
                      u, v, b, e, t, name, targets, relevant_fields)

    model_grid = grid_type(datapath; size=(1,1,Nz))

    # Return OneDimensionalTimeSeries with grid and variables coarse_grained to model resolution
    td_coarse = OneDimensionalTimeSeries(td, model_grid)

    return td_coarse

end

"""
    OneDimensionalTimeSeries(observation::OneDimensionalTimeSeries, grid)

Returns `observation::OneDimensionalTimeSeries` interpolated to `grid`.
"""
function OneDimensionalTimeSeries(data::OneDimensionalTimeSeries, grid::AbstractGrid)

    U = [ XFaceField(grid) for t in data.t ]
    V = [ YFaceField(grid) for t in data.t ]
    B = [ CenterField(grid) for t in data.t ]
    E = [ CenterField(grid) for t in data.t ]

    for i = 1:length(data.t)
        set!(U[i], data.u[i])
        set!(V[i], data.v[i])
        set!(B[i], data.b[i])
        set!(E[i], data.e[i])
    end

    return OneDimensionalTimeSeries(td.boundary_conditions,
                      grid,
                      data.constants,
                      data.diffusivities,
                      U,
                      V,
                      B,
                      E,
                      data.t,
                      data.name,
                      data.targets,
                      data.relevant_fields)
end

"""
    OneDimensionalTimeSeries(observation::OneDimensionalTimeSeries, grid)

Construct OneDimensionalTimeSeries from a time-series of Oceananigans LES data saved at `datapath`.
and interpolate the data to `grid`.
"""
function OneDimensionalTimeSeries(datapath, grid::AbstractGrid)
    td = OneDimensionalTimeSeries(datapath)
    return OneDimensionalTimeSeries(td, grid)
end