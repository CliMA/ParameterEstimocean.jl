
function variance(field::AbstractDataField{X, Y, Z, A, G, T, N} where {X, Y, Z, A, G <: RegularRectilinearGrid, T, N})

    # View of field.data that excludes halo points
    data = Oceananigans.Fields.interior(field)

    field_mean = mean(data)

    variance = zero(eltype(field))
    for j in eachindex(data)
        variance += (data[j] - field_mean)^2
    end

    # Average over the number of elements in the array
    return variance / length(data)
end

function profile_mean(data, field_name)
    total_mean = 0.0
    fields = getproperty(data, field_name)
    data = Oceananigans.Fields.interior(field)

    for target in data.targets
        field = fields[target]
        field_mean = mean(data)
        total_mean += field_mean
    end

    return total_mean / length(data.targets)
end

function max_variance(data::TruthData, field_name)
    maximum_variance = 0.0
    fields = getproperty(data, field_name)

    for target in data.targets
        field = fields[target]
        maximum_variance = max(maximum_variance, variance(field))
    end

    return maximum_variance
end

function mean_variance(data::TruthData, field_name)
    total_variance = 0.0
    fields = getproperty(data, field_name)

    for target in data.targets
        field = fields[target]
        total_variance += variance(field)
    end

    return total_variance / length(data.targets)
end

nan2inf(err) = isnan(err) ? Inf : err

function trapz(f, t)
    @inbounds begin
        integral = zero(eltype(t))
        for i = 2:length(t)
            integral += (f[i] + f[i-1]) * (t[i] - t[i-1])
        end
    end
    return integral
end


struct VarianceWeights{F, D, T, V}
       fields :: F
         data :: D
      targets :: T
    variances :: V
end

@inbounds normalize_variance(::Nothing, field, σ) = σ

function VarianceWeights(data; fields, targets=1:length(data), normalizer=nothing)
    variances = (; zip(fields, (zeros(length(targets)) for field in fields))...)

    for (k, field) in enumerate(fields)
        for i in 1:length(targets)
            @inbounds variances[k][i] = normalize_variance(normalizer, field, variance(data, field, i))
        end
    end

    return VarianceWeights(fields, data, targets, variances)
end

function simple_safe_save(savename, variable, name="calibration")

    temppath = savename[1:end-5] * "_temp.jld2"
    newpath = savename

    isfile(newpath) && mv(newpath, temppath, force=true)

    println("Saving to $savename...")
    save(newpath, name, variable)

    isfile(temppath) && rm(temppath)

    return nothing
end
