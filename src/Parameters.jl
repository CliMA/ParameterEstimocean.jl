module Parameters

using Oceananigans.Architectures: CPU, arch_array, architecture
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure
using Oceananigans.TurbulenceClosures: AbstractTimeDiscretization, ExplicitTimeDiscretization

using Printf
using Distributions

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
ln(X) ∼ N(μ, σ²),
```

where ``N(μ, σ²)`` is the `Normal` distribution with mean ``μ``
and variance ``σ²``.

The `mean` and variance ``s²`` (where ``s`` is the standard
deviation or `std`) are related to the parameters ``μ``
and ``σ²`` via

```math
m = exp{μ + σ² / 2}
s² = (exp{σ²} - 1) ⋅ m²
```

These formula allow us to calculate ``μ`` and ``σ`` given
``m`` and ``s²``, since rearranging the formula for ``s²``
gives

```math
exp{σ²} = m² / s² + 1
```

which then yields

```math
σ = sqrt(log(k)), 
```

where ``k = m² / s² + 1``. We then find that

```math
μ = log(m) - σ² / 2 .
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

"""
Y is the original constrained paramter, X is the normally-distributed, unconstrained parameter
with X ~ N(μ, σ)


Then

θ = L + (U - L) / 1 + exp(θ)
"""
struct ScaledLogitNormal{FT}
    μ :: FT
    σ :: FT
    lower_bound :: FT
    upper_bound :: FT
end

# From unconstrained to constrained
scaled_logit_normal_inverse_transform(L, U, θ) = L + (U - L) / (1 + exp(θ))

# From constrained to unconstrained
scaled_logit_normal_forward_transform(L, U, θ) = log((U - θ) / (θ - L))

"""
    ScaledLogitNormal(FT=Float64; bounds, midspread, center, width)

Return a `ScaledLogitNormal` distribution with parameters `μ`, `σ`, `lower_bound`
and `upper_bound`.
"""
function ScaledLogitNormal(FT=Float64;
                           bounds,
                           midspread = nothing,
                           center = nothing,
                           width = nothing)

    # Default
    μ = 0
    σ = 1
    L, U = bounds

    if !isnothing(center) || !isnothing(width)
        isnothing(center) && isnothing(width) ||
            throw(ArgumentError("Both center and width must be specified!"))

        isnothing(midspread) ||
            throw(ArgumentError("Cannot specify both midspread and (center, width)!"))

        midspread = [center - width/2, center + width/2]
    end

    if !isnothing(midspread)
        Li, Ui = midspread

        Li > L && Ui < U ||
            throw(ArgumentError("Midspread must lie within lower_bound and upper_bound."))

        # Compute lower and upper limits of midspread in unconstrained space
        L̃i = scaled_logit_normal_forward_transform(L, U, Li)
        Ũi = scaled_logit_normal_forward_transform(L, U, Ui)

        # Determine mean and std from unconstrained limits
        μ = (Ũi + L̃i) / 2
        σ = (Ũi - L̃i) / 1.349 # midspread interval spans 50% of total mass
    end

    return ScaledLogitNormal{FT}(μ, σ, L, U)
end

# Calculate the prior in unconstrained space given a prior in constrained space
unconstrained_prior(Π::LogNormal)         = Normal(1.0, Π.σ / Π.μ)
unconstrained_prior(Π::Normal)            = Normal(1.0, Π.σ / Π.μ)
unconstrained_prior(Π::ScaledLogitNormal) = Normal(Π.μ, Π.σ)

# Transform parameters from constrained (physical) space to unconstrained (EKI) space
transform_to_unconstrained(Π::Normal,    θ) = θ / abs(Π.μ)
transform_to_unconstrained(Π::LogNormal, θ) = log(θ) / abs(Π.μ)

transform_to_unconstrained(Π::ScaledLogitNormal, θ) =
    scaled_logit_normal_forward_transform(Π.lower_bound, Π.upper_bound, θ)

"""
    transform_to_constrained(Π, θ) = θ / Π.μ

Transform a parameter `θ` from unconstrained (EKI) space to constrained (physical) space,
given the _constrained_ prior distribution `Π`.
"""
transform_to_constrained(Π::Normal, θ)    = θ / Π.μ
transform_to_constrained(Π::LogNormal, θ) = exp(θ / Π.μ)

transform_to_constrained(Π::ScaledLogitNormal, θ) =
    scaled_logit_normal_inverse_transform(Π.lower_bound, Π.upper_bound, θ)

# Convenience vectorized version
transform_to_constrained(priors::NamedTuple, parameters::Vector) =
    NamedTuple(name => transform_to_constrained(priors[name], parameters[i])
               for (i, name) in enumerate(keys(priors)))

function inverse_covariance_transform(Π, parameters, covariance)
    diag = [covariance_transform_diagonal(Π[i], parameters[i]) for i=1:length(Π)]
    dT = Diagonal(diag)
    return dT * covariance * dT'
end

covariance_transform_diagonal(::LogNormal, p) = exp(p)
covariance_transform_diagonal(::Normal, p) = 1
covariance_transform_diagonal(Π::ScaledLogitNormal, p) = - (Π.upper_bound - Π.lower_bound) * exp(p) / (1 + exp(p))^2

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
julia> using OceanTurbulenceParameterEstimation.Parameters: closure_with_parameters

julia> struct ClosureSubModel; a; b end

julia> struct Closure; test; c end

julia> closure = Closure(ClosureSubModel(1, 2), 3)
Closure(ClosureSubModel(1, 2), 3)

julia> parameters = (a = 12, d = 7)
(a = 12, d = 7)

julia> closure_with_parameters(closure, parameters)
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

Use `parameters` to update closure from `closures` that corresponds to ensemble member `p_ensemble`.
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

function new_closure_ensemble(closures, θ)
    arch = architecture(closures)
    cpu_closures = arch_array(CPU(), closures)

    for (p, θp) in enumerate(θ)
        update_closure_ensemble_member!(cpu_closures, p, θp)
    end

    return arch_array(arch, cpu_closures)
end

end # module
