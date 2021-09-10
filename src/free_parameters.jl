using Oceananigans.Architectures: arch_array, architecture
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Printf

Base.show(io::IO, p::FreeParameters) = print(io, "$(typeof(p)):", '\n',
                                             @sprintf("% 24s: ", "parameter names"),
                                             (@sprintf("%-8s", n) for n in propertynames(p))..., '\n',
                                             @sprintf("% 24s: ", "values"),
                                             (@sprintf("%-8.4f", pᵢ) for pᵢ in p)...)

named_tuple_snippet(name) = Symbol(" $name,")

macro free_parameters(group_name, parameter_names...)
    N = length(parameter_names)
    
    function_elems = [named_tuple_snippet(name) for name in parameter_names]
    function_body = Symbol("(; ", function_elems..., ")")
    function_args = Symbol(function_elems...)

    return esc(quote
        # Positional argument constructor
        $group_name($function_args) = $function_body

        # Keyword argument constructor
        $group_name($function_args) = $function_body
    end)
end


#####
##### Setting parameters
#####

const ParameterValue = Union{Number, AbstractArray}

dict_properties(d::ParameterValue) = d

function dict_properties(d)
    p = Dict{Symbol, Any}(n => dict_properties(getproperty(d, n)) for n in propertynames(d))
    p[:type] = typeof(d)
    return p
end

construct_object(d::ParameterValue, parameters; name=nothing) = kw ∈ keys(parameters) ? getproperty(parameters, kw) : d

function construct_object(specification_dict, parameters; name=nothing, type_parameter=nothing)
    type = Constructor = specification_dict[:type]
    kwargs_vector = [construct_object(specification_dict[name], parameters; name) for name in fieldnames(type) if name != :type]
    return isnothing(type_parameter) ? Constructor(kwargs_vector...) : Constructor{type_parameter}(kwargs_vector...)
end

"""
    closure_with_parameters(closure, parameters)

Returns a new object where for each (parameter_name, parameter_value) pair 
in `parameters`, the value corresponding to the key in `object` that matches
`parameter_name` is replaced with `parameter_value`.

Example:

```julia-repl
julia> struct ClosureSubModel a; b end

julia> struct Closure test; c end

julia> closure = Closure(ClosureSubModel(1, 2), 3)
Closure(ClosureSubModel(1, 2), 3)

julia> parameters = (a = 12, c = 7)
(a = 12, c = 7)

julia> closure_with_parameters(closure, parameters)
Closure(ClosureSubModel(12, 2), 7)
```
"""
closure_with_parameters(closure::AbstractTurbulenceClosure{TD}, parameters) where TD =
    construct_object(dict_properties(closure), parameters; type_parameter=TD)
