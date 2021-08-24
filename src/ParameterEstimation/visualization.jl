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


"""
    visualize_realizations(data, model, params...)

Visualize the data alongside several realizations of `column_model`
for each set of parameters in `params`.
"""
function visualize_realizations(model, data_batch, parameters::FreeParameters;
                                                 fields = [:b, :u, :v, :e],
                                                 filename = "realizations.png"
                                )
        
    fig = Figure(resolution = (200*(length(fields)+1), 200*length(data_batch)), font = "CMU Serif")
    colors = [:black, :red, :blue]

    function empty_plot!(fig_position)
        ax = fig_position = Axis(fig_position)
        hidedecorations!(ax)
        hidespines!(ax, :t, :b, :l, :r)
    end

    for (i, data) in enumerate(data_batch)

        targets = data.targets
        snapshots = round.(Int, range(targets[1], targets[end], length=3))
        bcs = data.boundary_conditions

        empty_plot!(fig[i,1])
        text!(fig[i,1], "Qᵇ = $(tostring(bcs.Qᵇ)) m⁻¹s⁻³\nQᵘ = $(tostring(bcs.Qᵘ)) m⁻¹s⁻²\nf = $(tostring(data.constants[:f])) s⁻¹", 
                    position = (0, 0), 
                    align = (:center, :center), 
                    textsize = 15,
                    justification = :left)

        model_predictions = model_time_series(parameters, model, data)

        for (j, field) in enumerate(fields)
            middle = j > 1 && j < length(fields)
            remove_spines = j == 1 ? (:t, :r) : j == length(fields) ? (:t, :l) : (:t, :l, :r)
            axis_position = j == length(fields) ? (ylabelposition=:right, yaxisposition=:right) : NamedTuple()

            j += 1 # reserve the first column for row labels

            info = field_guide[field]

            # field data for each time step
            truth = getproperty(data, field)
            prediction = getproperty(model_predictions, field)

            z = field ∈ [:u, :v] ? data.grid.zF[1:data.grid.Nz] : data.grid.zC[1:data.grid.Nz]

            to_plot = field ∈ data.relevant_fields

            if to_plot

                ax = Axis(fig[i,j]; xlabelpadding=0, xtickalign=1, ytickalign=1, 
                                            merge(axis_position, info.axis_args)...)

                hidespines!(ax, remove_spines...)

                middle && hideydecorations!(ax, grid=false)

                lins = []
                for (color_index, target) in enumerate(snapshots)
                    l = lines!([interior(truth[target]) .* info.scaling ...], z; color = (colors[color_index], 0.4))
                    push!(lins, l)
                    l = lines!([interior(prediction[target - snapshots[1] + 1]) .* info.scaling ...], z; color = (colors[color_index], 1.0), linestyle = :dash)
                    push!(lins, l)
                end

                times = @. round((data.t[snapshots] - data.t[snapshots[1]]) / 86400, sigdigits=2)
                Legend(fig[1,3:4], lins, [["LES, t = $time days" for time in times]; ["Model, t = $time days" for time in times]], nbanks=2)
                lins = []
            else
                
                empty_plot!(fig[i,j])
            end
        end
    end

    save(filename, fig, px_per_unit = 2.0)
end

function visualize_and_save(ce, parameters, directory; fields=[:b, :u, :v, :e])

        # o = open_output_file(directory*"/result.txt")
        # write(o, "Training relative weights: $(ce.calibration.relative_weights) \n")
        # write(o, "Validation relative weights: $(ce.validation.relative_weights) \n")
        # write(o, "Training default parameters: $(ce.validation.default_parameters) \n")
        # write(o, "Validation default parameters: $(ce.validation.default_parameters) \n")

        # write(o, "------------ \n \n")
        # default_parameters = ce.default_parameters
        # train_loss_default = ce.calibration.loss(default_parameters)
        # valid_loss_default = ce.validation.loss(default_parameters)
        # write(o, "Default parameters: $(default_parameters) \nLoss on training: $(train_loss_default) \nLoss on validation: $(valid_loss_default) \n------------ \n \n")

        # train_loss = ce.calibration.loss(parameters)
        # valid_loss = ce.validation.loss(parameters)
        # write(o, "Parameters: $(parameters) \nLoss on training: $(train_loss) \nLoss on validation: $(valid_loss) \n------------ \n \n")

        # write(o, "Training loss reduction: $(train_loss/train_loss_default) \n")
        # write(o, "Validation loss reduction: $(valid_loss/valid_loss_default) \n")
        # close(o)

        parameters = ce.parameters.ParametersToOptimize(parameters)

        for dataset in [ce.calibration, ce.validation]

            all_data = dataset.data_batch
            model = dataset.model
            set!(model, parameters)

            for data_length in Set(length.(all_data))

                data_batch = [d for d in all_data if length(d) == data_length]
                days = data_batch[1].t[end]/86400

                visualize_realizations(model, data_batch, parameters;
                                                 fields = fields,
                                                 filename = joinpath(directory, "$(days)_day_simulations.png"))
            end
        end
end
