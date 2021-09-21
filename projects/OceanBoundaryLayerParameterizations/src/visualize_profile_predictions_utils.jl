function open_output_file(directory)
        isdir(directory) || mkpath(directory)
        file = joinpath(directory, "output.txt")
        touch(file)
        o = open(file, "w")
        return o
end

function writeout(o, name, loss, params)
        param_vect = [params...]
        loss_value = loss(params)
        write(o, "----------- \n")
        write(o, "$(name) \n")
        write(o, "Parameters: $(param_vect) \n")
        write(o, "Loss: $(loss_value) \n")
        saveplot(params, name, loss)
end

field_guide = Dict(
    :u => (
        axis_args = (ylabel="z (m)", xlabel="U velocity (dm/s)"),
        scaling = 1e1,
    ),

    :v => (
        axis_args = (xlabel="V velocity (dm/s)",),
        scaling = 1e1,
    ),

    :b => (
        axis_args = (xlabel="Buoyancy (cN/kg)",),
        scaling = 1e2,
    ),

    :e => (
        axis_args = (ylabel="z (m)", xlabel="TKE (cm²/s²)"),
        scaling = 1e4,
    )
)

function tostring(num)
    num == 0 && return "0"
    om = Int(floor(log10(abs(num))))
    num /= 10.0^om
    num = num%1 ≈ 0 ? Int(num) : round(num; digits=2)
    return "$(num)e$om"
end
