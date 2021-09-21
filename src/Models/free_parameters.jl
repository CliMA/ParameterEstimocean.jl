using Oceananigans.Architectures: arch_array, architecture

abstract type FreeParameters{N, T} <: FieldVector{N, T} end

Base.show(io::IO, p::FreeParameters) = print(io, "$(typeof(p)):", '\n',
                                             @sprintf("% 24s: ", "parameter names"),
                                             (@sprintf("%-8s", n) for n in propertynames(p))..., '\n',
                                             @sprintf("% 24s: ", "values"),
                                             (@sprintf("%-8.4f", pᵢ) for pᵢ in p)...)

macro free_parameters(GroupName, parameter_names...)
    N = length(parameter_names)
    parameter_exprs = [:($name :: T; ) for name in parameter_names]
    return esc(quote
        Base.@kwdef mutable struct $GroupName{T} <: FreeParameters{$N, T}
            $(parameter_exprs...)
        end
    end)
end

function get_free_parameters(closure::AbstractTurbulenceClosure)
    paramnames = Dict()
    paramtypes = Dict()
    kw_params = Dict() # for parameters that are not contained in structs but rather as explicit keyword arguments in `m.closure`
    for pname in propertynames(closure) # e.g. :surface_TKE_flux
        p = getproperty(closure, pname) # e.g. p = TKESurfaceFlux{Float64}(3.62, 1.31)

        # if pname ∈ [:dissipation_parameter, :mixing_length_parameter]
        if p isa Number
            kw_params[pname] = p #e.g. kw_params[:Cᴰ] = 2.91

        # if pname ∈ [:surface_TKE_flux, :diffusivity_scaling]
        elseif all(fieldtypes(typeof(p)) .<: Number)
            paramnames[pname] = propertynames(p) #e.g. paramnames[:surface_TKE_flux] = (:Cᵂu★, :CᵂwΔ)
            paramtypes[pname] = typeof(p) #e.g. paramtypes[:surface_TKE_flux] = TKESurfaceFlux{Float64}

        end
    end

    return paramnames, paramtypes, kw_params
end
 
function DefaultFreeParameters(closure::AbstractTurbulenceClosure, freeparamtype)
    paramnames, paramtypes, kw_params = get_free_parameters(closure)
    #e.g. paramnames[:surface_TKE_flux] = (:Cᵂu★, :CᵂwΔ);
    #     paramtypes[:surface_TKE_flux] = TKESurfaceFlux{Float64}
    #     kw_params[:Cᴰ] = 2.91

    alldefaults = (ptype() for ptype in values(paramtypes))

    freeparams = [] # list of parameter values in the order specified by fieldnames(freeparamtype)
    for pname in fieldnames(freeparamtype) # e.g. :Cᵂu★
        for ptype in alldefaults # e.g. TKESurfaceFlux{Float64}(3.62, 1.31)
            pname ∈ propertynames(ptype) && (push!(freeparams, getproperty(ptype, pname)); break)
            pname ∈ keys(kw_params) && (push!(freeparams, kw_params[pname]); break)
        end
    end

    return eval(Expr(:call, freeparamtype, freeparams...)) # e.g. ParametersToOptimize([1.0,2.0,3.0])
end

#####
##### new_closure
#####

function new_closure(closure::AbstractTurbulenceClosure, free_parameters)

    paramnames, paramtypes, kw_params = get_free_parameters(closure)
    #e.g. paramnames[:surface_TKE_flux] = (:Cᵂu★, :CᵂwΔ)
    #e.g. paramtypes[:surface_TKE_flux] = TKESurfaceFlux{Float64}
    #e.g. kw_params[:Cᴰ] = 2.91

    # All keyword arguments to be passed in when defining the new closure
    new_closure_kwargs = kw_params

    # Populate paramdicts with the new values for each parameter name `pname` under `ptypename`
    for ptypename in keys(paramtypes) # e.g. :diffusivity_scaling, :surface_TKE_flux

        existing_parameters = getproperty(closure, ptypename)

        new_ptype_kwargs = Dict()

        for pname in propertynames(existing_parameters)

            p = pname ∈ propertynames(free_parameters) ?
                    getproperty(free_parameters, pname) :
                    getproperty(existing_parameters, pname)

            new_ptype_kwargs[pname] = p
        end

        # Create new parameter struct for `ptypename` with parameter values given by `new_ptype_kwargs`
        new_closure_kwargs[ptypename] = paramtypes[ptypename](; new_ptype_kwargs...)
    end

    # Include closure properties that do not correspond to model parameters, if any
    for ptypename in propertynames(closure)

        if ptypename ∉ keys(new_closure_kwargs)
            new_closure_kwargs[ptypename] = getproperty(closure, ptypename)
        end

    end

    ClosureType = typeof(closure)
    args = [new_closure_kwargs[x] for x in fieldnames(ClosureType)]
    new_closure = ClosureType(args...)

    return new_closure
end

function new_closure(closure::AbstractArray, free_parameters::Vector{<:FreeParameters})
    arch = architecture(closure)
    closure = Array(closure)
    Ex, Ey = size(closure)
    closure = [new_closure(closure[i, j], free_parameters[i]) for i=1:Ex, j=1:Ey]
    return arch_array(arch, closure)
end

new_closure(closure::AbstractArray, free_parameters::FreeParameters) =
    new_closure(closure, [free_parameters for i=1:size(closure, 1)])

#####
##### set!
#####

function set!(model::EnsembleModel, free_parameters)
    model.closure = new_closure(model.closure, free_parameters)
    return nothing
end
