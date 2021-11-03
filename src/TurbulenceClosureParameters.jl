module TurbulenceClosureParameters

using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Printf

#####
##### Free parameters
#####

struct FreeParameters{N, P}
     names :: N
    priors :: P
end

"""
    FreeParameters(priors; names = Symbol.(keys(priors)))

Return named `FreeParameters` with priors.
Free parameter `names` are inferred from the keys of `priors` if not provided.

Example
=======

```jldoctest
julia> using Distributions, OceanTurbulenceParameterEstimation

julia> priors = (ν = Normal(1e-4, 1e-5), κ = Normal(1e-3, 1e-5))
(ν = Normal{Float64}(μ=0.0001, σ=1.0e-5), κ = Normal{Float64}(μ=0.001, σ=1.0e-5))

julia> free_parameters = FreeParameters(priors)
FreeParameters with 2 parameters
├── names: (:ν, :κ)
└── priors: Dict{Symbol, Any}
    ├── ν => Normal{Float64}(μ=0.0001, σ=1.0e-5)
    └── κ => Normal{Float64}(μ=0.001, σ=1.0e-5)
```
"""
function FreeParameters(priors; names = Symbol.(keys(priors)))
    priors = NamedTuple(name => priors[name] for name in names)
    return FreeParameters(Tuple(names), priors)
end

free_parameters_str(fp::FreeParameters) = "$(fp.names)"

function prior_show(io, priors, name, prefix, width)
    print(io, @sprintf("%s %s => ", prefix, lpad(name, width, " ")))
    show(io, priors[name])
    return nothing
end

function Base.show(io::IO, p::FreeParameters)
    Np = length(p)
    print(io, "FreeParameters with $Np parameters", '\n',
              "├── names: $(p.names)", '\n',
              "└── priors: Dict{Symbol, Any}")

    maximum_name_length = maximum([length(string(name)) for name in p.names]) 

    for (i, name) in enumerate(p.names)
        prefix = i == length(p.names) ? "    └──" : "    ├──"
        print(io, '\n')
        prior_show(io, p.priors, name, prefix, maximum_name_length)
    end
    
    return nothing
end

Base.length(p::FreeParameters) = length(p.names)

#####
##### Setting parameters
#####

const ParameterValue = Union{Number, AbstractArray}

dict_properties(d::ParameterValue) = d

function dict_properties(d)
    p = Dict{Symbol, Any}(n => dict_properties(getproperty(d, n)) for n in propertynames(d))
    p[:type] = typeof(d).name.wrapper
    return p
end

construct_object(d::ParameterValue, parameters; name=nothing) = name ∈ keys(parameters) ? getproperty(parameters, name) : d

function construct_object(specification_dict, parameters; name=nothing, type_parameter=nothing)
    type = Constructor = specification_dict[:type]
    kwargs_vector = [construct_object(specification_dict[name], parameters; name) for name in fieldnames(type) if name != :type]
    return isnothing(type_parameter) ? Constructor(kwargs_vector...) : Constructor{type_parameter}(kwargs_vector...)
end

"""
    closure_with_parameters(closure, parameters)

Returns a new object where for each (`parameter_name`, `parameter_value`) pair 
in `parameters`, the value corresponding to the key in `object` that matches
`parameter_name` is replaced with `parameter_value`.

Example
=======

```jldoctest
julia> using OceanTurbulenceParameterEstimation

julia> struct ClosureSubModel; a; b end

julia> struct Closure; test; c end

julia> closure = Closure(ClosureSubModel(1, 2), 3)
Closure(ClosureSubModel(1, 2), 3)

julia> parameters = (a = 12, d = 7)
(a = 12, d = 7)

julia> OceanTurbulenceParameterEstimation.TurbulenceClosureParameters.closure_with_parameters(closure, parameters)
Closure(ClosureSubModel(12, 2), 7)
```
"""
closure_with_parameters(closure, parameters) = construct_object(dict_properties(closure), parameters)

closure_with_parameters(closure::AbstractTurbulenceClosure{TD}, parameters) where TD =
    construct_object(dict_properties(closure), parameters; type_parameter=TD)

update_closure_ensemble_member!(closures::AbstractVector, p_ensemble, parameters) =
    closures[p_ensemble] = closure_with_parameters(closures[p_ensemble], parameters)

function update_closure_ensemble_member!(closures::AbstractMatrix, p_ensemble, parameters)
    for j in size(closures, 2) # Assume that ensemble varies along first dimension
        closures[p_ensemble, j] = closure_with_parameters(closures[p_ensemble, j], parameters)
    end
    return nothing
end

end # module
