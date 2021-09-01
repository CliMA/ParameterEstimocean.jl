
## Utils for writing results to output file

function open_output_file(directory)
        isdir(directory) || mkpath(directory)
        file = directory*"output.txt"
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