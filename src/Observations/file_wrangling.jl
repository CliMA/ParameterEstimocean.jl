function get_iterations(datapath)
    file = jldopen(datapath, "r")
    iters = parse.(Int, keys(file["timeseries/t"]))
    sort!(iters) # We always want iterations to be sorted from small to large
    close(file)
    return iters
end

function get_times(datapath)
    iters = get_iterations(datapath)
    times = zeros(length(iters))

    jldopen(datapath, "r") do file
        for (i, iter) in enumerate(iters)
            times[i] = file["timeseries/t/$iter"]
        end
    end

    @assert issorted(times) "Simulation data i"

    return times
end

function get_parameter(filename, group, parameter_name, default=nothing)
    parameter = default

    jldopen(filename) do file
        if parameter_name ∈ keys(file["$group"])
            parameter = file["$group/$parameter_name"]
        end
    end

    return parameter
end

function get_data(varname, datapath, iter; reversed=false)
    file = jldopen(datapath, "r")
    var = file["timeseries/$varname/$iter"]
    close(file)

    # Drop extra singleton dimensions if they exist
    if ndims(var) > 1
        droplist = []
        for d = 1:ndims(var)
           size(var, d) == 1 && push!(droplist, d)
       end
       var = dropdims(var, dims=Tuple(droplist))
    end

    reversed && reverse!(var)

    return var
end

function new_field(fieldtype, simulation_grid, data)

    newfield = fieldtype(simulation_grid)

    # Reshape `data` to the size of `newfield`'s interior
    d = reshape(data, size(newfield))

    # Sets the interior of `newfield` to values of `data`
    newfield .= d

    return newfield
end