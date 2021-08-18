#####
##### ParameterizedModel
#####

mutable struct ParameterizedModel{M<:AbstractModel, T}
    model :: M
       Δt :: T
end

#
# Accessing model fields
#

Base.getproperty(m::ParameterizedModel, ::Val{:Δt}) = getfield(m, :Δt)
Base.getproperty(m::ParameterizedModel, ::Val{:model}) = getfield(m, :model)
Base.getproperty(m::ParameterizedModel, p::Symbol) = getproperty(m, Val(p))
# Base.getproperty(m::ParameterizedModel, ::Val{p}) where p = getproperty(m.model, p)

function Base.getproperty(m::ParameterizedModel, ::Val{p}) where p

    p ∈ propertynames(m.model.tracers) && return m.model.tracers[p]

    p ∈ propertynames(m.model.velocities) && return m.model.velocities[p]

    return getproperty(m.model, p)

end

#
# Iterating the model
#

time(model) = model.clock.time
iteration(model) = model.clock.iteration

"""
    time_step!(model, Δt, Nt)
Evolve `model` for `Nt` time steps with time-step `Δt`.
"""
function time_step!(model, Δt, Nt)
    for step = 1:Nt
        time_step!(model, Δt)
    end
    return nothing
end

"""
    run_until!(model, Δt, tfinal)
Run `model` until time `tfinal` with time-step `Δt`.
"""
function run_until!(model, Δt, tfinal)
    Nt = floor(Int, (tfinal - time(model))/Δt)
    time_step!(model, Δt, Nt)

    last_Δt = tfinal - time(model)
    last_Δt == 0 || time_step!(model, last_Δt)

    return nothing
end

run_until!(pm::ParameterizedModel, time) = run_until!(pm.model, pm.Δt, time)
