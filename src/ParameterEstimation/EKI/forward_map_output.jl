#
# z-score normalization
#

function get_normalization_functions(loss::LossFunction)

    normalize_function = Dict()

    for data in loss.data_batch
        case = data.name
        normalize_function[case] = Dict()
        fields = data.relevant_fields
        targets = data.targets

        for field in fields
            μ = OceanTurbulenceParameterEstimation.profile_mean(data, field; targets=targets, indices=data_indices)
            σ = mean_std(data, field; targets=targets, indices=data_indices)

            normalize(Φ) = (Φ .- μ) ./ σ
            normalize_function[case][field] = normalize
        end
    end
    return normalize_function
end


#
# Forward map output
#

abstract type AbstractForwardMapOutput end

"""
In the unidimensional formulation of EKI, we let the forward map output G be the square root of the
evaluation of the loss function on θ.
"""
struct SqrtLossForwardMapOutput{P} <: AbstractForwardMapOutput
    prior :: P
    SqrtLossForwardMapOutput(dataset, prior::P) where P = new{P}(prior)
end

# Loss function minimum
observation(G::SqrtLossForwardMapOutput) = [0.0]

(G::SqrtLossForwardMapOutput)(u) = @. sqrt(loss(transform_unconstrained_to_constrained(G.prior, u)))

"""
In this multidimensional formulation of EKI, we let the forward map output G compute the concatenated final 
profiles for the predicted `u`, `v`, `b`, and `e` at the final timestep. Thus the truth y corresponds to the 
concatenated final profiles of the ground truth simulation data.
"""
struct ConcatenatedProfilesForwardMapOutput{DS, NF, PR} <: AbstractForwardMapOutput
    dataset :: DS
    normalize_functions :: NF
    prior :: PR

    function ConcatenatedProfilesForwardMapOutput(dataset::DS, prior::PR) where {DS, PR}
        normalize_functions = get_normalization_functions(dataset)
        NF = typeof(normalize_functions)
        return new{DS, NF, PR}(dataset, normalize_functions, prior)
    end
end

# Concatenated profiles at the final timestep
function (G::ConcatenatedProfilesForwardMapOutput)(u)
    all = []
    parameters = transform_unconstrained_to_constrained(G.prior, u)
    outputs = model_time_series(G.dataset, parameters)
    data_batch = G.dataset.data_batch
    for (dataindex, data, output) in zip(eachindex(data_batch), data_batch, outputs)
        last = data.targets[end]
        for fieldname in data.relevant_fields
            model_field = getproperty(output, fieldname)[last]
            zscore_normalize = G.normalize_functions[data.name][fieldname]
            model_profile = zscore_normalize(interior(model_field))
            push!(all, model_profile...)
        end
    end
    return hcat(all...)
end

# Concatenated profiles at the final timestep according to 
function observation(G::ConcatenatedProfilesForwardMapOutput)
    y  = []
    for data in G.dataset.data_batch
        for fieldname in data.relevant_fields
            last = data.targets[end]
            data_field = getproperty(data, fieldname)[last]
            zscore_normalize = normalize_function[data.name][fieldname]
            obs = zscore_normalize(interior(data_field))
            push!(y, obs...)
        end
    end
    return y
end
