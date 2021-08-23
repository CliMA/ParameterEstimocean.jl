function get_iterations(datapath)
    file = jldopen(datapath, "r")
    iters = parse.(Int, keys(file["timeseries/t"]))
    close(file)
    return iters
end

function get_times(datapath)
    iters = get_iterations(datapath)
    t = zeros(length(iters))
    jldopen(datapath, "r") do file
        for (i, iter) in enumerate(iters)
            t[i] = file["timeseries/t/$iter"]
        end
    end
    return t
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

function get_grid_params(datapath::String)
    file = jldopen(datapath, "r")

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]
    Nz = file["grid/Nz"]

    Lx = file["grid/Lx"]
    Ly = file["grid/Ly"]
    Lz = file["grid/Lz"]

    close(file)
    return Nx, Ny, Nz, Lx, Ly, Lz
end