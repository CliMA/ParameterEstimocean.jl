using Dao

Dao.optimal(result::Dao.AnnealingProblem) = optimal(result.markov_chains[end])

OceanTurbulenceParameterEstimation.visualize_realizations(result::Dao.AnnealingProblem, params; kwargs...) =
       visualize_realizations(result.negative_log_likelihood.model,
                              result.negative_log_likelihood.data,
                              result.negative_log_likelihood.loss.targets[[1, end]], params; kwargs...)

function optimum_series(problem, parameter)
    chains = problem.markov_chains
    return [getproperty(optimal(chain).param, parameter) for chain in chains]
end

function optimum_series(problem)
    ParameterType = typeof(problem.markov_chains[1][1].param).name.wrapper
    optimums = [optimum_series(problem, p) for p in fieldnames(ParameterType)]
    return ParameterType(optimums...)
end

function restart_extend_and_save!(calibration, chunks, path)

    loss = calibration.negative_log_likelihood
    previous_chain = calibration.markov_chains[end]
    covariance_estimate = cov(previous_chain)
    initial_link = optimal(previous_chain)
    sampler = MetropolisSampler(calibration.perturbation(covariance_estimate,
                                                         calibration.perturbation_args...))

    # L₀ = scale * chain[1]
    # Lnew = scale * optimal(chain)
    # => Lnew = L₀ / chain[1] * optimal(chain)
    loss.scale *= optimal(previous_chain).error / previous_chain[1].error

    println("Starting the first chain with chunksize $(chunks[1])...")
    @time new_chain = MarkovChain(chunks[1], initial_link, loss, sampler)
    push!(calibration.markov_chains, new_chain)

    status(new_chain)
    simple_safe_save(path, calibration)

    for chunk in chunks[2:end]
        @time extend!(new_chain, chunk)
        status(new_chain)
        simple_safe_save(path, calibration)
    end

    return nothing
end

function extend_and_save!(calibration, chunks, path)

    chain = calibration.markov_chains[end]

    println("Extending a Markov chain...")

    status(chain)

    for chunk in chunks
        @time extend!(chain, chunk)
        status(chain)
        simple_safe_save(path, calibration)
    end

    return nothing
end

function continuation(calibration, nearby_calibration, chunks, continuation_path)

    previous_nearby_chain = nearby_calibration.markov_chains[end]
    Cᵢ = optimal(previous_nearby_chain).param

    # Re-estimate covariance
    calibration_chain = calibration.markov_chains[end]
    loss = calibration.negative_log_likelihood
    covariance_estimate = cov(calibration_chain)
    initial_link = MarkovLink(loss, Cᵢ)

    # Re-annealing
    continued_calibration = anneal(loss, Cᵢ, covariance_estimate, calibration.perturbation,
                                   calibration.perturbation_args...;
                                               samples = 4000, #calibration.samples,
                                            iterations = 3, #calibration.iterations,
                                    annealing_schedule = calibration.annealing_schedule,
                                   covariance_schedule = calibration.covariance_schedule)

    extend_and_save!(continued_calibration, chunks, continuation_path)

    return continued_calibration
end

function continuation(child, parent_calibration, chunks; suffix="continuation")
    child_path = path(child)
    child_calibration = load_calibration(child_path)

    continuation_name = child[1:end-5] * "-$suffix.jld2"
    continuation_path = path(continuation_name)

    @show child_path continuation_path

    child_continuation = continuation(child_calibration, parent_calibration,
                                      chunks, continuation_path)

    return child_continuation
end

get_position(ax) = [b for b in ax.get_position().bounds]

function shift_up!(ax, shift)
    pos = get_position(ax)
    pos[2] += shift
    ax.set_position(pos)
    return nothing
end

function shift_down!(ax, shift)
    pos = get_position(ax)
    pos[2] -= shift
    ax.set_position(pos)
    return nothing
end

function shift_right!(ax, shift)
    pos = get_position(ax)
    pos[1] += shift
    ax.set_position(pos)
    return nothing
end

function shift_left!(ax, shift)
    pos = get_position(ax)
    pos[1] -= shift
    ax.set_position(pos)
    return nothing
end

function stretch_x!(ax, stretch)
    pos = get_position(ax)
    pos[3] += stretch
    ax.set_position(pos)
    return nothing
end

function stretch_y!(ax, stretch)
    pos = get_position(ax)
    pos[4] += stretch
    ax.set_position(pos)
    return nothing
end

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
