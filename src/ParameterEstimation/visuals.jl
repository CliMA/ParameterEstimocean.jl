function visualize_and_save(ce, parameters, directory; include_tke=false)

        o = open_output_file(directory*"/result.txt")
        write(o, "Training relative weights: $(ce.calibration.relative_weights) \n")
        write(o, "Validation relative weights: $(ce.validation.relative_weights) \n")
        write(o, "Training default parameters: $(ce.validation.default_parameters) \n")
        write(o, "Validation default parameters: $(ce.validation.default_parameters) \n")

        write(o, "------------ \n \n")
        default_parameters = ce.default_parameters
        train_loss_default = ce.calibration.nll(default_parameters)
        valid_loss_default = ce.validation.nll(default_parameters)
        write(o, "Default parameters: $(default_parameters) \n")
        write(o, "Loss on training: $(train_loss_default) \n")
        write(o, "Loss on validation: $(valid_loss_default) \n")

        write(o, "------------ \n \n")
        parameters = [parameters...]
        train_loss = ce.calibration.nll_wrapper(parameters)
        valid_loss = ce.validation.nll_wrapper(parameters)
        write(o, "Parameters: $(parameters) \n")
        write(o, "Loss on training: $(train_loss) \n")
        write(o, "Loss on validation: $(valid_loss) \n")

        write(o, "------------ \n \n")
        write(o, "Training loss reduction: $(train_loss/train_loss_default) \n")
        write(o, "Validation loss reduction: $(valid_loss/valid_loss_default) \n")
        close(o)

        function get_Δt(Nt)
                Δt = 90
                if Nt > 400; Δt = 240; end
                if Nt > 800; Δt = 360; end
                return Δt
        end

        path = directory*"/Plots/"
        mkpath(path)

        parameters = ce.parameters.ParametersToOptimize(parameters)
        function helper(ce, dataset)
                for LEScase in values(dataset.LESdata)
                        nll, _ = get_nll(LEScase, ce.parameters, dataset.relative_weights)
                        OceanTurbulenceParameterEstimation.set!(nll.model, parameters)
                        Nt = length(nll.data)

                        fields = include_tke ? nll.loss.fields : (f for f in nll.loss.fields if f != :e)
                        p = visualize_realizations(nll.model, nll.data, 60:get_Δt(Nt):length(nll.data), parameters, fields = nll.loss.fields)
                        PyPlot.savefig(path*"$(Nt)_$(nll.data.name).png")
                end
        end

        helper(ce, ce.calibration)
        helper(ce, ce.validation)
end
