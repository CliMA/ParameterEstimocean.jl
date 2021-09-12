dict_properties(d::Union{Number, AbstractArray}) = d

function dict_properties(d)
    p = Dict{Symbol, Any}(n => dict_properties(getproperty(d, n)) for n in propertynames(d))
    p[:type] = typeof(d).name.wrapper
    return p
end

build(d::Union{Number, AbstractArray}, parameters; kw=nothing) = kw ∈ keys(parameters) ? getproperty(parameters, kw) : d

function build(d, parameters; kw=nothing)
    type = d[:type]
    kwargs_vector = [build(d[kw], parameters; kw) for kw in fieldnames(type) if kw != :type]
    return type(kwargs_vector...)
end

# parameters is a NamedTuple
new_closure(closure, parameters) = build(dict_properties(closure), parameters)

#
# Example
#

julia> struct Test a; b end

julia> struct OuterTest test; c end

julia> o = OuterTest(Test(1, 2), 3)
OuterTest(Test(1, 2), 3)

julia> parameters = (a = 12, c = 7)
(a = 12, c = 7)

julia> new_closure(o, parameters)
OuterTest(Test(12, 2), 7)