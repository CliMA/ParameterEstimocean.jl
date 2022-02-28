module Parameters

export FreeParameters, lognormal, ScaledLogitNormal

using Oceananigans.Architectures: CPU, arch_array, architecture
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Oceananigans.TurbulenceClosures: AbstractTimeDiscretization, ExplicitTimeDiscretization

using Printf
using Distributions
using LinearAlgebra

using SpecialFunctions: erfinv
using Distributions: AbstractRNG, ContinuousUnivariateDistribution

#####
##### Priors
#####

"""
    lognormal(; mean, std)

Return `Lognormal` distribution parameterized by 
the distribution `mean` and standard deviation `std`.

Notes
=====

A variate `X` is `LogNormal` distributed if

```math
\\log(X) ∼ 𝒩(μ, σ²) ,
```

where ``𝒩(μ, σ²)`` is the `Normal` distribution with mean ``μ``
and variance ``σ²``.

The `mean` and variance ``s²`` (where ``s`` is the standard
deviation or `std`) are related to the parameters ``μ``
and ``σ²`` via

```math
 m = \\exp(μ + σ² / 2),
```
```math
s² = [\\exp(σ²) - 1] m².
```

These formula allow us to calculate ``μ`` and ``σ`` given
``m`` and ``s²``, since rearranging the formula for ``s²``
gives

```math
\\exp(σ²) = m² / s² + 1
```

which then yields

```math
σ = \\sqrt{\\log(m² / s² + 1)}.
```

We then find that

```math
μ = \\log(m) - σ² / 2 .
```

See also
[wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution#Generation_and_parameters).
"""
function lognormal(; mean, std)
    k = std^2 / mean^2 + 1 # intermediate variable
    σ = sqrt(log(k))
    μ = log(mean) - σ^2 / 2
    return LogNormal(μ, σ)
end

struct ScaledLogitNormal{T} <: ContinuousUnivariateDistribution
    μ :: T
    σ :: T
    lower_bound :: T
    upper_bound :: T

    ScaledLogitNormal{T}(μ, σ, L, U) where T = new{T}(T(μ), T(σ), T(L), T(U))
end

"""Return a logit-normally distributed variate given the normally-distributed variate `X`."""
normal_to_scaled_logit_normal(L, U, X) = L + (U - L) / (1 + exp(X))

"""Return a normally-distributed variate given the logit-normally distributed variate `Y`."""
scaled_logit_normal_to_normal(L, U, Y) = log((U - Y) / (Y - L))

Base.rand(rng::AbstractRNG, d::ScaledLogitNormal) =
    normal_to_scaled_logit_normal(d.lower_bound, d.upper_bound, rand(rng, Normal(d.μ, d.σ)))

unit_normal_std(mass) = 1 / (2 * √2 * erfinv(mass))

"""
    ScaledLogitNormal([FT=Float64;] bounds=(0, 1), mass=0.5, interval=nothing)

Return a `ScaledLogitNormal` distribution with compact support within `bounds`.

`interval` is an optional 2-element tuple or Array. When specified,
the parameters `μ` and `σ` of the underlying `Normal` distribution
are calculated so that `mass` fraction of the probability density
lies within `interval`.

If `interval` is not specified, then `μ=0` and `σ=1` by default.

Notes
=====

`ScaledLogitNormal` is a four-parameter distribution
generated by the transformation

```math
Y = L + (U - L) / [1 + \\exp(X)],
```

of the normally-distributed variate ``X ∼ 𝒩(μ, σ)``. The four parameters
governing the distribution of ``Y`` are thus

- ``L``:  lower bound (0 for the `LogitNormal` distribution)
- ``U``:  upper bound (1 for the `LogitNormal` distribution)
- ``μ``:  mean of the underlying `Normal` distribution
- ``σ²``: variance of the underlying `Normal` distribution
"""
function ScaledLogitNormal(FT=Float64; bounds=(0, 1), mass=0.5, interval=nothing, μ=nothing, σ=nothing)
    L, U = bounds

    if isnothing(interval) # use default μ=0 and σ=1 if not set
        isnothing(μ) && (μ = 0)
        isnothing(σ) && (σ = 1)

    elseif !isnothing(interval) # try to compute μ and σ

        Li, Ui = interval

        # User friendliness
        (!isnothing(μ) || !isnothing(σ)) && @warn "Using interval and mass to determine μ and σ."
        0 < mass < 1 || throw(ArgumentError("Mass must lie between 0 and 1."))
        Li > L && Ui < U || throw(ArgumentError("Interval limits must lie between `bounds`."))

        # Compute lower and upper limits of midspread in unconstrained space
        #
        # Note that the _lower_ bound in unconstrained space is associated with the
        # _upper_ bound in constrained space, and vice versa.
        L̃i = scaled_logit_normal_to_normal(L, U, Ui)
        Ũi = scaled_logit_normal_to_normal(L, U, Li)
        
        μ = (Ũi + L̃i) / 2

        # Note that the mass beneath a half-width `δ` of the
        # standard Normal distribution is
        #
        # mass = 2 / √(2π) ∫₀ᵟ exp(-x^2 / 2) dx
        #      = erf(δ / √2)
        #
        # For an `interval = (Ũi, L̃i)` of the normal distribution,
        # the non-dimensional half-width is
        #
        # δ = (Ũi - L̃i) / 2σ
        #
        # where σ is the distribution's standard deviation.
        # We then find
        #
        # erfinv(mass) = (Ũi - L̃i) / (2 * √2 * σ) ,
        #
        # and rearranging to solve for σ yields
        # 
        σ = (Ũi - L̃i) / (2 * √2 * erfinv(mass))
    end

    return ScaledLogitNormal{FT}(μ, σ, L, U)
end

# Calculate the prior in unconstrained space given a prior in constrained space
unconstrained_prior(Π::LogNormal)         = Normal(Π.μ / abs(Π.μ), Π.σ / abs(Π.μ))
unconstrained_prior(Π::Normal)            = Normal(Π.μ / abs(Π.μ), Π.σ / abs(Π.μ))
unconstrained_prior(Π::ScaledLogitNormal) = Normal(Π.μ, Π.σ)

"""
    transform_to_unconstrained(Π, Y)

Transform the "constrained" (physical) variate `Y` into it's
unconstrained (normally-distributed) counterpart `X` through the
forward map associated with `Π`.

If some mapping between ``Y`` and the normally-distributed ``X`` is
defined via

```math
Y = g(X).
```

Then `transform_to_unconstrained` is the inverse ``X = g^{-1}(Y)``.
The change of variables ``g(X)`` determines the distribution `Π` of `Y`.

Example
=======

The logarithm of a `LogNormal(μ, σ)` distributed variate is normally-distributed,
such that the forward trasform ``f ≡ \\exp``,

```math
Y = \\exp(X),
```

and the inverse trasnform is the natural logarithm ``f^{-1} ≡ \\log``,

```math
\\log(Y) = X ∼ 𝒩(μ, σ).
```
"""
transform_to_unconstrained(Π::Normal,    Y) = Y / abs(Π.μ)
transform_to_unconstrained(Π::LogNormal, Y) = log(Y^(1 / abs(Π.μ))) # log(Y) / abs(Π.μ)

transform_to_unconstrained(Π::ScaledLogitNormal, Y) =
    scaled_logit_normal_to_normal(Π.lower_bound, Π.upper_bound, Y)

"""
    transform_to_constrained(Π, X)

Transform an "unconstrained", normally-distributed variate `X`
to "constrained" (physical) space via the map associated with
the distribution `Π` of `Y`.
"""
transform_to_constrained(Π::Normal, X)    = X * abs(Π.μ)
transform_to_constrained(Π::LogNormal, X) = exp(X * abs(Π.μ))

transform_to_constrained(Π::ScaledLogitNormal, X) =
    normal_to_scaled_logit_normal(Π.lower_bound, Π.upper_bound, X)

# Convenience vectorized version
transform_to_constrained(priors::NamedTuple, X::AbstractVector) =
    NamedTuple(name => transform_to_constrained(priors[name], X[i])
               for (i, name) in enumerate(keys(priors)))

# Convenience matrixized version assuming particles vary on 2nd dimension
transform_to_constrained(priors::NamedTuple, X::AbstractMatrix) =
    [transform_to_constrained(priors, X[:, k]) for k = 1:size(X, 2)]

function inverse_covariance_transform(Π, X, covariance)
    diag = [covariance_transform_diagonal(Π[i], X[i]) for i=1:length(Π)]
    dT = Diagonal(diag)
    return dT * covariance * dT'
end

covariance_transform_diagonal(::LogNormal, X) = exp(X)
covariance_transform_diagonal(::Normal, X) = 1
covariance_transform_diagonal(Π::ScaledLogitNormal, X) = - (Π.upper_bound - Π.lower_bound) * exp(X) / (1 + exp(X))^2

#####
##### Free parameters
#####

"""
    struct FreeParameters{N, P}

A container for free parameters that includes the parameter names and their
corresponding prior distributions.
"""
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

Base.summary(fp::FreeParameters) = "$(fp.names)"

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

"""
    construct_object(specification_dict, parameters; name=nothing, type_parameter=nothing)
    
    construct_object(d::ParameterValue, parameters; name=nothing)

Return a composite type object whose properties are prescribed by the `specification_dict`
dictionary. All parameter values are given the values in `specification_dict` *unless* they
are included as a parameter name-value pair in the named tuple `parameters`, in which case
the value in `parameters` is asigned.

The `construct_object` is recursively called upon every property that is included in `specification_dict`
until a property with a numerical value is reached. The object's constructor name must be
included in `specification_dict` under key `:type`.

Example
=======

```jldoctest; filter = [r".*Dict{Symbol.*", r".*:type       => Closure.*", r".*:c          => 3.*", r".*:subclosure => Dict{Symbol.*"]
julia> using OceanTurbulenceParameterEstimation.Parameters: construct_object, dict_properties, closure_with_parameters

julia> struct Closure; subclosure; c end

julia> struct ClosureSubModel; a; b end

julia> sub_closure = ClosureSubModel(1, 2)
ClosureSubModel(1, 2)

julia> closure = Closure(sub_closure, 3)
Closure(ClosureSubModel(1, 2), 3)

julia> specification_dict = dict_properties(closure)
Dict{Symbol, Any} with 3 entries:
  :type       => Closure
  :c          => 3
  :subclosure => Dict{Symbol, Any}(:a=>1, :b=>2, :type=>ClosureSubModel)

julia> new_closure = construct_object(specification_dict, (a=2.1,))
Closure(ClosureSubModel(2.1, 2), 3)
  
julia> another_new_closure = construct_object(specification_dict, (b=π, c=2π))
Closure(ClosureSubModel(1, π), 6.283185307179586)
```

"""
construct_object(d::ParameterValue, parameters; name=nothing) =
    name ∈ keys(parameters) ? getproperty(parameters, name) : d

function construct_object(specification_dict, parameters;
                          name=nothing, type_parameter=nothing)

    type = Constructor = specification_dict[:type]
    kwargs_vector = [construct_object(specification_dict[name], parameters; name)
                        for name in fieldnames(type) if name != :type]

    return isnothing(type_parameter) ? Constructor(kwargs_vector...) : 
                                       Constructor{type_parameter}(kwargs_vector...)
end

"""
    dict_properties(object)

Return a dictionary with all properties of an `object` and their values, including the 
`object`'s type name. If any of the `object`'s properties is not a numerical value but
instead a composite type, then `dict_properties` is called recursively on that `object`'s
property returning a dictionary with all properties of that composite type. Recursion
ends when properties of type `ParameterValue` are found.
"""
function dict_properties(object)
    p = Dict{Symbol, Any}(n => dict_properties(getproperty(object, n)) for n in propertynames(object))
    p[:type] = typeof(object).name.wrapper

    return p
end

dict_properties(object::ParameterValue) = object

"""
    closure_with_parameters(closure, parameters)

Return a new object where for each (`parameter_name`, `parameter_value`) pair 
in `parameters`, the value corresponding to the key in `closure` that matches
`parameter_name` is replaced with `parameter_value`.

Example
=======

Create a placeholder `Closure` type that includes a parameter `c` and a sub-closure
with two parameters: `a` and `b`. Then construct a closure with values `a, b, c = 1, 2, 3`.

```jldoctest closure_with_parameters
julia> struct Closure; subclosure; c end

julia> struct ClosureSubModel; a; b end

julia> sub_closure = ClosureSubModel(1, 2)
ClosureSubModel(1, 2)

julia> closure = Closure(sub_closure, 3)
Closure(ClosureSubModel(1, 2), 3)
```

Providing `closure_with_parameters` with a named tuple of parameter names and values,
and a recursive search in all types and subtypes within `closure` is done and whenever
a parameter is found whose name exists in the named tuple we provided, its value is 
then replaced with the value provided.

```jldoctest closure_with_parameters
julia> new_parameters = (a = 12, d = 7)
(a = 12, d = 7)

julia> using OceanTurbulenceParameterEstimation.Parameters: closure_with_parameters

julia> closure_with_parameters(closure, new_parameters)
Closure(ClosureSubModel(12, 2), 3)
```
"""
closure_with_parameters(closure, parameters) = construct_object(dict_properties(closure), parameters)

closure_with_parameters(closure::AbstractTurbulenceClosure{ExplicitTimeDiscretization}, parameters) =
    construct_object(dict_properties(closure), parameters, type_parameter=nothing)

closure_with_parameters(closure::AbstractTurbulenceClosure{TD}, parameters) where {TD <: AbstractTimeDiscretization} =
    construct_object(dict_properties(closure), parameters; type_parameter=TD)

closure_with_parameters(closures::Tuple, parameters) =
    Tuple(closure_with_parameters(closure, parameters) for closure in closures)

"""
    update_closure_ensemble_member!(closures, p_ensemble, parameters)

Use `parameters` to update the `p_ensemble`-th closure from and array of `closures`.
The `p_ensemble`-th closure corresponds to ensemble member `p_ensemble`.
"""
update_closure_ensemble_member!(closure, p_ensemble, parameters) = nothing

update_closure_ensemble_member!(closures::AbstractVector, p_ensemble, parameters) =
    closures[p_ensemble] = closure_with_parameters(closures[p_ensemble], parameters)

function update_closure_ensemble_member!(closures::AbstractMatrix, p_ensemble, parameters)
    for j in 1:size(closures, 2) # Assume that ensemble varies along first dimension
        closures[p_ensemble, j] = closure_with_parameters(closures[p_ensemble, j], parameters)
    end
    
    return nothing
end

function update_closure_ensemble_member!(closure_tuple::Tuple, p_ensemble, parameters)
    for closure in closure_tuple
        update_closure_ensemble_member!(closure, p_ensemble, parameters)
    end
    return nothing
end

"""
    new_closure_ensemble(closures, θ, arch=CPU())

Return a new set of `closures` in which all closures that have free parameters are updated.
Closures with free parameters are expected as `AbstractArray` of `TurbulenceClosures`, and
this allows `new_closure_ensemble` to go through all closures in `closures` and only update
the parameters for the any closure that is of type `AbstractArray`. The `arch`itecture
(`CPU()` or `GPU()`) defines whethere `Array` or `CuArray` is returned.
"""
function new_closure_ensemble(closures::AbstractArray, θ, arch)
    cpu_closures = arch_array(CPU(), closures)

    for (p, θp) in enumerate(θ)
        update_closure_ensemble_member!(cpu_closures, p, θp)
    end

    return arch_array(arch, cpu_closures)
end
new_closure_ensemble(closures::Tuple, θ, arch) = 
    Tuple(new_closure_ensemble(closure, θ, arch) for closure in closures)

new_closure_ensemble(closure, θ, arch) = closure

end # module
