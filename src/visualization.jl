styles = ("--", ":", "-.", "o-", "^--")
defaultcolors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Default kwargs for plot routines
default_modelkwargs = Dict(:linewidth=>2, :alpha=>0.8)
default_datakwargs = Dict(:linewidth=>3, :alpha=>0.6)
default_legendkwargs = Dict(:fontsize=>10, :loc=>"lower right", :frameon=>true, :framealpha=>0.5)

removespine(side) = gca().spines[side].set_visible(false)
removespines(sides...) = [removespine(side) for side in sides]

function plot_data!(axs, data, targets, fields; datastyle="-", datakwargs...)
    for (iplot, i) in enumerate(targets)
        lbl = iplot == 1 ? "LES, " : ""
        lbl *= @sprintf("\$ t = %0.2f \$ hours", data.t[i]/hour)

        for (ipanel, field) in enumerate(fields)
            sca(axs[ipanel])
            dfld = getproperty(data, field)[i]
            plot(dfld, datastyle; label=lbl, color=defaultcolors[iplot], datakwargs...)
        end
    end
    return nothing
end

function label_ax!(ax, field)
    if field === :U
        sca(ax)
        xlabel("\$ U \$ velocity \$ \\mathrm{(m \\, s^{-1})} \$")
    end

    if field === :V
        sca(ax)
        xlabel("\$ V \$ velocity \$ \\mathrm{(m \\, s^{-1})} \$")
    end

    if field === :T
        sca(ax)
        xlabel("Temperature (Celsius)")
    end

    if field === :S
        sca(ax)
        xlabel("Salinity (psu)")
    end

    if field === :e
        sca(ax)
        xlabel("\$ e \$ \$ \\mathrm{(m^2 \\, s^{-2})} \$")
    end

    return nothing
end


function format_axs!(axs, fields; legendkwargs...)
    sca(axs[1])
    removespines("top", "right")
    ylabel(L"z \, \mathrm{(meters)}")
    legend(; legendkwargs...)

    for iax in 2:length(axs)-1
        sca(axs[iax])
        removespines("top", "right", "left")
        axs[iax].tick_params(left=false, labelleft=false)
    end

    if length(fields) > 1
        sca(axs[end])
        axs[end].yaxis.set_label_position("right")
        axs[end].tick_params(left=false, labelleft=false, right=true, labelright=true)
        removespines("top", "left")
        ylabel(L"z \, \mathrm{(meters)}")
    end

    [label_ax!(ax, fields[i]) for (i, ax) in enumerate(axs)]

    return nothing
end

"""
    visualize_realizations(data, model, params...)

Visualize the data alongside several realizations of `column_model`
for each set of parameters in `params`.
"""
function visualize_realizations(column_model, column_data, targets, params::FreeParameters...;
                                                    fig = nothing,
                                                figsize = (10, 4),
                                plot_first_model_target = false,
                                            paramlabels = ["" for p in params], datastyle="-",
                                            modelkwargs = Dict(),
                                             datakwargs = Dict(),
                                           legendkwargs = Dict(),
                                                 fields = (:U, :V, :T)
                                )

    # Merge defaults with user-specified options
     modelkwargs = merge(default_modelkwargs, modelkwargs)
      datakwargs = merge(default_datakwargs, datakwargs)
    legendkwargs = merge(default_legendkwargs, legendkwargs)

    #
    # Make plot
    #

    if fig === nothing
        fig, axs = subplots(ncols=length(fields), figsize=figsize, sharey=true)
    else
        axs = fig._get_axes()
        for ax in axs
            sca(ax)
            cla()
        end
    end

    for (iparam, param) in enumerate(params)
        set!(column_model, param)
        set!(column_model, column_data, targets[1])

        for (iplot, i) in enumerate(targets)
            run_until!(column_model.model, column_model.Δt, column_data.t[i])

            if iplot == length(targets)
                lbl =  @sprintf("%s ParameterizedModel, \$ t = %0.2f \$ hours",
                                paramlabels[iparam], column_data.t[i]/hour)
            else
                lbl = ""
            end

            if iplot > 1 || plot_first_model_target
                for (ipanel, field) in enumerate(fields)
                    sca(axs[ipanel])
                    model_field = getproperty(column_model.model, field)
                    plot(model_field, styles[iparam]; color=defaultcolors[iplot], label=lbl, modelkwargs...)
                end
            end
        end
    end

    plot_data!(axs, column_data, targets, fields; datastyle=datastyle, datakwargs...)
    format_axs!(axs, fields; legendkwargs...)

    return fig, axs
end

function plot_loss_function(loss, model, data, params...;
                            labels=["Parameter set $i" for i = 1:length(params)],
                            time_norm=:second)

    numerical_time_norm = eval(time_norm)

    fig, axs = subplots()

    for (i, param) in enumerate(params)
        evaluate!(loss, param, model, data)
        plot(loss.time_series.time / numerical_time_norm, loss.time_series.data, label=labels[i])
    end

    removespines("top", "right")

    time_units = string(time_norm, "s")
    xlabel("Time ($time_units)")
    ylabel("Time-resolved loss function")
    legend()

    return fig, axs
end

function calculate_error!(error, model, data)
    set!(error, data)
    for i in eachindex(error)
        @inbounds error[i] = (model[i] - data[i])^2
    end
    return nothing
end

function visualize_loss_function(loss, model, data, target_index, params...;
                                 labels=["Parameter set $i" for i = 1:length(params)],
                                 figsize=(10, 4),
                                 legendkwargs=Dict())

    target = loss.targets[target_index]
    ϕerror = loss.profile.discrepency
    legendkwargs = merge(default_legendkwargs, legendkwargs)

    # Some shenanigans so things like 'enumerate' work good.
    fields = loss.fields isa Symbol ? (loss.fields,) : loss.fields

    fig, axs = subplots(nrows=2, ncols=length(fields), figsize=figsize, sharey=true)

    for (iparam, param) in enumerate(params)
        initialize_forward_run!(model, data, param, loss.targets[1])
        run_until!(model.model, model.Δt, data.t[target])
        evaluate!(loss, param, model, data)

        for (i, field) in enumerate(fields)
            ϕmodel = getproperty(model, field)
            ϕdata = getproperty(data, field)[target]
            calculate_discrepency!(loss.profile, ϕmodel, ϕdata)

            if iparam == 1
                sca(axs[1, i])
                plot(ϕdata; linestyle="-", color=defaultcolors[iparam])
            end

            sca(axs[1, i])
            plot(ϕmodel; linestyle="--", color=defaultcolors[iparam],
                    label="ParameterizedModel, " * labels[iparam])

            sca(axs[2, i])
            error_label = @sprintf("Loss = %.2e, %s", loss.profile.analysis(loss.profile.discrepency),
                                   labels[iparam])

            plot(ϕerror; linestyle="-", color=defaultcolors[iparam], label=error_label)
        end
    end

    pause(0.1)
    format_axs!(axs[1, :], loss.fields; legendkwargs...)

    pause(0.1)
    format_axs!(axs[2, :], loss.fields; legendkwargs...)

    return nothing
end

function visualize_markov_chain!(ax, chain, parameter; after=1, bins=100, alpha=0.6, density=true,
                                 facecolor="b")

    parameters = propertynames(chain[1].param)
    samples = Dao.params(chain, after=after)

    C = map(x->getproperty(x, parameter), samples)

    sca(ax)
    ρ, _, _ = plt.hist(C, bins=bins, alpha=alpha, density=density, facecolor=facecolor)
    removespines("left", "right", "top")
    ax.tick_params(left=false, labelleft=false)

    ρmax = maximum(ρ)

    C_optimal = getproperty(optimal(chain).param, parameter)
    C_median = median(C)
    C_mean = mean(C)

    plot(C_optimal, 1.2ρmax, "*"; mfc="None", mec=facecolor, linestyle="-", markersize=8)
    plot(C_median , 1.2ρmax, "o"; mfc="None", mec=facecolor, linestyle="-", markersize=8)
    plot(C_mean   , 1.2ρmax, "^"; mfc="None", mec=facecolor, linestyle="-", markersize=8)

    pause(0.1)

    return ρ
end

function visualize_markov_chain!(chain; figsize=(8, 12), parameter_latex_guide=nothing, kwargs...)
    nparameters = length(chain[1].param)
    ρ = []
    fig, axs = subplots(nrows=nparameters, figsize=figsize)

    for (i, p) in enumerate(propertynames(chain[1].param))
        ax = axs[i]
        ρᵢ = visualize_markov_chain!(ax, chain, p; kwargs...)
        push!(ρ, ρᵢ)
        parameter_latex_guide != nothing && xlabel(parameter_latex_guide[p])
    end

    return fig, axs, ρ
end
