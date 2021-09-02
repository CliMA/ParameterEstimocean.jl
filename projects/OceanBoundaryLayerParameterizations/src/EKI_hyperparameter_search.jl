
#
# Hyperparameter search
#

function loss_reduction(calibration, validation, initial_parameters, kwargs)
    params, losses, mean_vars = eki(calibration, initial_parameters; kwargs...)
    valid_loss_start = validation(initial_parameters)
    valid_loss_final = validation(params)

    @info "parameters: $params"
    @info "final loss / initial loss: $(losses[end] / losses[1])"

    train_loss_reduction = losses[end] / losses[1]
    valid_loss_reduction = valid_loss_final / valid_loss_start
    return train_loss_reduction, valid_loss_reduction
end

## Stds within bounds

function plot_stds_within_bounds(calibration, validation, initial_parameters, directory; xrange=-3:0.25:5)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for stds_within_bounds in xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (stds_within_bounds = stds_within_bounds,))
        loss_reductions[stds_within_bounds] = train_loss_reduction
        val_loss_reductions[stds_within_bounds] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus prior std's within bounds (n_σ)", ylabel="Loss reduction (Final / Initial)", xlabel="prior std's spanned by bound width (n_σ)", legend=false, lw=3)
    plot!(loss_reductions, label="training", color=:purple, lw=4)
    plot!(val_loss_reductions, label="validation", color=:blue, lw=4)
    Plots.savefig(p, directory*"stds_within_bounds.pdf")
    println("loss-minimizing stds within bounds: $(argmin(val_loss_reductions))")
end

## Prior Variance
function plot_prior_variance(calibration, validation, initial_parameters, directory; xrange=0.1:0.1:1.0)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for variance = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (stds_within_bounds = variance,))
        loss_reductions[variance] = train_loss_reduction
        val_loss_reductions[variance] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus prior variance", ylabel="Loss reduction (Final / Initial)", xlabel="Prior variance")
    plot!(loss_reductions, label="training", lw=4, color=:purple)
    plot!(val_loss_reductions, label="validation", lw=4, color=:blue)
    Plots.savefig(p, directory*"variance.pdf")
    v = argmin(var_val_loss_reductions)
    println("loss-minimizing variance: $(v)")
    return v
end

## Number of ensemble members

function plot_num_ensemble_members(calibration, validation, initial_parameters, directory; xrange=1:5:30)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for N_ens = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (N_ens = N_ens,))
        loss_reductions[N_ens] = train_loss_reduction
        val_loss_reductions[N_ens] = val_loss_reduction
    end
    p = Plots.plot(title="Loss reduction versus N_ens", xlabel="N_ens", ylabel="Loss reduction (Final / Initial)", legend=false, lw=3)
    plot!(loss_reductions, label="training")
    plot!(val_loss_reductions, label="validation")
    Plots.savefig(p, directory*"N_ens.pdf")
end

## Observation noise level

function plot_observation_noise_level(calibration, validation, initial_parameters, directory; xrange=-2.0:0.2:3.0)
    loss_reductions = Dict()
    val_loss_reductions = Dict()
    for log_noise_level = xrange
        train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (noise_level = 10.0^log_noise_level,))
        loss_reductions[log_noise_level] = train_loss_reduction
        val_loss_reductions[log_noise_level] = val_loss_reduction
    end
    p = Plots.plot(title="Loss Reduction versus Observation Noise Level", xlabel="log₁₀(Observation noise level)", ylabel="Loss reduction (Final / Initial)", legend=:topleft)
    plot!(loss_reductions, label="training", lw=4, color=:purple)
    plot!(val_loss_reductions, label="validation", lw=4, color=:blue)
    Plots.savefig(p, directory*"obs_noise_level.pdf")
    nl = argmin(val_loss_reductions)
    println("loss-minimizing obs noise level: $(nl)")
    return 10^nl
end

function plot_prior_variance_and_obs_noise_level(calibration, validation, initial_parameters, directory; vrange=0.40:0.025:0.90, nlrange=-2.5:0.1:0.5)
    Γθs = collect(vrange)
    Γys = 10 .^ collect(nlrange)
    losses = zeros((length(Γθs), length(Γys)))
    counter = 1
    countermax = length(Γθs)*length(Γys)
    for i in 1:length(Γθs)
        for j in 1:length(Γys)
            println("progress $(counter)/$(countermax)")
            Γθ = Γθs[i]
            Γy = Γys[j]
            train_loss_reduction, val_loss_reduction = loss_reduction(calibration, validation, initial_parameters, (stds_within_bounds=Γθ, noise_level=Γy, N_iter=10, N_ens=10, informed_priors=false))
            losses[i, j] = val_loss_reduction
            counter += 1
        end
    end
    p = Plots.heatmap(Γys, Γθs, losses, xlabel=L"\Gamma_y", ylabel=L"\Gamma_\theta", size=(250,250), yscale=:log10)
    Plots.savefig(p, directory*"GammaHeatmap.pdf")
    v = Γθs[argmin(losses)[1]]
    nl = Γys[argmin(losses)[2]]
    println("loss-minimizing Γθ: $(v)")
    println("loss-minimizing log10(Γy): $(log10(nl))")
    return v, nl
end
