using OceanTurbulenceParameterEstimation.InverseProblems: vectorize, transpose_model_output

"""
    visualize!(output::ForwardMap;
                    fields = [:u, :v, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png"
                    )

    For visualizing 1-dimensional time series predictions.
"""
function visualize!(ip::InverseProblem, parameters;
                    fields = [:u, :v, :b, :e],
                    directory = pwd(),
                    filename = "realizations.png"
                    )

    isdir(directory) || makedir(directory)

    observations = vectorize(ip.observations)

    forward_run!(ip, parameters)

    # Vector of OneDimensionalTimeSeries objects, one for each observation
    predictions = transpose_model_output(time_series_collector, observations)
        
    fig = Figure(resolution = (200*(length(fields)+1), 200*length(ip.observations)), font = "CMU Serif")
    colors = [:black, :red, :blue]

    function empty_plot!(fig_position)
        ax = fig_position = Axis(fig_position)
        hidedecorations!(ax)
        hidespines!(ax, :t, :b, :l, :r)
    end

    # function get_data!()

    for (i, observation) in enumerate(observations)

        prediction = predictions[i]

        targets = observation.times
        snapshots = round.(Int, range(targets[1], targets[end], length=3))
        
        Qᵇ = ip.simulation.model.tracers.b.boundary_conditions.top.condition[1,i]
        Qᵘ = ip.simulation.model.velocities.u.boundary_conditions.top.condition[1,i]
        f = ip.simulation.model.coriolis[1,i].f

        empty_plot!(fig[i,1])
        text!(fig[i,1], "Qᵇ = $(tostring(Qᵇ)) m⁻¹s⁻³\nQᵘ = $(tostring(Qᵘ)) m⁻¹s⁻²\nf = $(tostring(f)) s⁻¹", 
                    position = (0, 0), 
                    align = (:center, :center), 
                    textsize = 15,
                    justification = :left)

        for (j, field) in enumerate(fields)
            middle = j > 1 && j < length(fields)
            remove_spines = j == 1 ? (:t, :r) : j == length(fields) ? (:t, :l) : (:t, :l, :r)
            axis_position = j == length(fields) ? (ylabelposition=:right, yaxisposition=:right) : NamedTuple()

            j += 1 # reserve the first column for row labels

            info = field_guide[field]

            # field data for each time step
            truth = getproperty(data, field)
            prediction = getproperty(predictions_batch[i], field)

            z = field ∈ [:u, :v] ? data.grid.zᵃᵃᶠ[1:data.grid.Nz] : data.grid.zᵃᵃᶜ[1:data.grid.Nz]

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

                legendlabel(time) = ["LES, t = $time days", "Model, t = $time days"]
                Legend(fig[1,2:3], lins, vcat([legendlabel(time) for time in times]...), nbanks=2)
                lins = []
            else
                
                empty_plot!(fig[i,j])
            end
        end
    end

    save(joinpath(directory, filename), fig, px_per_unit = 2.0)
    return nothing
end

visualize!(ip::InverseProblem, parameters; kwargs...) = visualize!(model_time_series(ip, parameters); kwargs...)

function visualize_and_save!(calibration, validation, parameters, directory; fields=[:u, :v, :b, :e])
        isdir(directory) || makedir(directory)

        path = joinpath(directory, "results.txt")
        o = open_output_file(path)
        write(o, "Training relative weights: $(calibration.relative_weights) \n")
        write(o, "Validation relative weights: $(validation.relative_weights) \n")
        write(o, "Training default parameters: $(validation.default_parameters) \n")
        write(o, "Validation default parameters: $(validation.default_parameters) \n")

        write(o, "------------ \n \n")
        default_parameters = calibration.default_parameters
        train_loss_default = calibration(default_parameters)
        valid_loss_default = validation(default_parameters)
        write(o, "Default parameters: $(default_parameters) \nLoss on training: $(train_loss_default) \nLoss on validation: $(valid_loss_default) \n------------ \n \n")

        train_loss = calibration(parameters)
        valid_loss = validation(parameters)
        write(o, "Parameters: $(parameters) \nLoss on training: $(train_loss) \nLoss on validation: $(valid_loss) \n------------ \n \n")

        write(o, "Training loss reduction: $(train_loss/train_loss_default) \n")
        write(o, "Validation loss reduction: $(valid_loss/valid_loss_default) \n")
        close(o)

        parameters = calibration.parameters.ParametersToOptimize(parameters)

        for inverse_problem in [calibration, validation]

            all_data = inverse_problem.observations
            simulation = inverse_problem.simulation
            set!(simulation.model, parameters)

            for data_length in Set(length.(getproperty.(all_data, :t)))

                observations = [d for d in all_data if length(d.t) == data_length]
                days = observations[1].t[end]/86400

                new_ip = InverseProblem()

                visualize!(simulation, observations, parameters;
                            fields = fields,
                            filename = joinpath(directory, "$(days)_day_simulations.png"))
            end
        end
    
    return nothing
end
