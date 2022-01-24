using CairoMakie

# Vector of NamedTuples, ensemble mean at each iteration
ensemble_means(eki) = getproperty.(eki.iteration_summaries, :ensemble_mean)

# N_param x N_iter matrix, ensemble covariance at each iteration
ensemble_std(eki) = sqrt.(hcat(diag.(getproperty.(eki.iteration_summaries, :ensemble_cov))...))

parameter_names(eki) = eki.inverse_problem.free_parameters.names

function plot_parameter_convergence!(eki, directory; true_parameters=nothing, n_columns=3)

    means = ensemble_means(eki)
    θθ_std_arr = ensemble_std(eki)

    pnames = parameter_names(eki)
    N_param = length(pnames)
    N_iter = length(eki.iteration_summaries) - 1 # exclude 0th element

    n_rows = Int(ceil(N_param / n_columns))
    ax_coords = [(i, j) for i = 1:n_rows, j = 1:n_columns]

    fig = Figure(resolution = (500n_columns, 200n_rows))
    for (i, pname) in enumerate(pnames)
        coords = ax_coords[i]
        ax = Axis(fig[coords...],
            xlabel = "Iteration",
            xticks = 0:N_iter,
            ylabel = string(pname))
        ax.ylabelsize = 20

        mean_values = [getproperty.(means, pname)...]
        lines!(ax, 0:N_iter, mean_values)
        band!(ax, 0:N_iter, mean_values .+ θθ_std_arr[i, :], mean_values .- θθ_std_arr[i, :])
        isnothing(true_parameters) || hlines!(ax, [true_parameters[pname]], color = :red)
    end

    save(joinpath(directory, "parameter_convergence.pdf"), fig)
end

function plot_pairwise_ensembles!(eki, directory, true_parameters=nothing)

    pnames = parameter_names(eki)
    N_param = length(pnames)
    N_iter = length(eki.iteration_summaries) - 1 # exclude 0th element
    for (i1, pname1) in enumerate(pnames), (i2, pname2) in enumerate(pnames)
        if i1 < i2

            f = Figure()
            axtop = Axis(f[1, 1])
            axmain = Axis(f[2, 1], xlabel = string(pname1), ylabel = string(pname2))
            axright = Axis(f[2, 2])
            scatters = []
            for iteration in [0, 1, 2, N_iter]
                ensemble = eki.iteration_summaries[iteration].parameters
                ensemble = [[particle[pname1], particle[pname2]] for particle in ensemble]
                ensemble = transpose(hcat(ensemble...)) # N_ensemble x 2
                push!(scatters, scatter!(axmain, ensemble))
                density!(axtop, ensemble[:, 1])
                density!(axright, ensemble[:, 2], direction = :y)
            end
            isnothing(true_parameters) || begin
                vlines!(axmain,  [true_parameters[pname1]], color = :red)
                vlines!(axtop,   [true_parameters[pname1]], color = :red)
                hlines!(axmain,  [true_parameters[pname2]], color = :red)
                hlines!(axright, [true_parameters[pname2]], color = :red)
            end
            colsize!(f.layout, 1, Fixed(300))
            colsize!(f.layout, 2, Fixed(200))
            rowsize!(f.layout, 1, Fixed(200))
            rowsize!(f.layout, 2, Fixed(300))
            Legend(f[1, 2], scatters,
                ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration $N_iter"],
                position = :lb)
            hidedecorations!(axtop, grid = false)
            hidedecorations!(axright, grid = false)
            xlims!(axright, 0, 10)
            ylims!(axtop, 0, 10)
            save(joinpath(directory, "pairwise_ensembles_$(pname1)_$(pname2).pdf"), f)
        end
    end
end

function plot_error_convergence!(eki, directory, true_parameters=nothing)

    means = ensemble_means(eki)
    N_iter = length(eki.iteration_summaries) - 1 # exclude 0th element
    y = observation_map(eki.inverse_problem)

    f = Figure()
    output_distances = [mapslices(norm, (forward_map(eki.inverse_problem, [means...])[:, 1:(N_iter+1)] .- y), dims = 1)...]
    lines(f[1, 1], 0:N_iter, output_distances, color = :blue, linewidth = 2,
        axis = (title = "Output distance",
            xlabel = "Iteration",
            ylabel = "|G(θ̅ₙ) - y|",
            yscale = log10))
    
    isnothing(true_parameters) || begin
        weight_distances = [norm(collect(means[iter]) .- collect(true_parameters)) for iter = 0:N_iter]
        lines(f[1, 2], 0:N_iter, weight_distances, color = :red, linewidth = 2,
            axis = (title = "Parameter distance",
                xlabel = "Iteration",
                ylabel = "|θ̅ₙ - θ⋆|",
                yscale = log10))
    end

    save(joinpath(directory, "error_convergence_summary.png"), f);
end