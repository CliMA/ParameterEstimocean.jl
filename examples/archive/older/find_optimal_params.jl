using Statistics, Distributions, PyPlot, Plots
using OceanTurb, OceanTurbulenceParameterEstimation, Dao
using OceanTurbulenceParameterEstimation.TKEMassFluxOptimization
using OceanTurbulenceParameterEstimation.TKEMassFluxOptimization: ParameterizedModel
using Optim

function writeout(o, name, params)
        param_vect = [params...]
        loss_value = loss(params)
        write(o, "----------- \n")
        write(o, "$(name) \n")
        write(o, "Parameters: $(param_vect) \n")
        write(o, "Loss: $(loss_value) \n")
        saveplot(params, name)
end

prefix = "/Users/adelinehillier/.julia/dev/"
suffix = "instantaneous_statistics.jld2"
files = ["8DayLinearStrat/general_strat_4/",
         "8DayLinearStrat/general_strat_8/",
         "8DayLinearStrat/general_strat_16/",
         "8DayLinearStrat/general_strat_32/",
         "4DaySuite/free_convection/",
         "4DaySuite/strong_wind/",
         "4DaySuite/strong_wind_no_rotation/",
         "4DaySuite/strong_wind_weak_cooling/",
         "4DaySuite/weak_wind_strong_cooling/",
         ]
files = [prefix*x*suffix for x in files]
general_strat = files[1:4]
four_day_suite = files[5:end]

mymodel = build_model(files[4]);
include("my_models.jl")
build_model = tke_free_convection_independent_diffusivities
# build_model = conv_adj_independent_diffusivites

directory = pwd() * "/find_optimal_params/$(build_model)/"
isdir(directory) || mkpath(directory)

for file in files

        mymodel = build_model(file);
        tdata = mymodel.tdata
        model = mymodel.model
        ParametersToOptimize = mymodel.ParametersToOptimize
        default_parameters = mymodel.default_parameters
        bounds = mymodel.bounds
        loss_function, loss = build_loss(model, tdata)

        f = directory*"calibrate_$(tdata.name)/output.txt"
        touch(f)
        o = open(f, "w")
        write(o, "Calibrating $(tdata.name) scenario \n")
        write(o, "Default parameters: $(default_parameters) \n")
        writeout(o, "Default", default_parameters)

        @info "Running Iterative Simulated Annealing..."

        variance = Array(ParametersToOptimize((0.1 * bound[2] for bound in bounds)...))
        prob = anneal(loss, default_parameters, variance, BoundedNormalPerturbation, bounds;
                     iterations = 10,
                        samples = 1000,
             annealing_schedule = AdaptiveExponentialSchedule(initial_scale=100.0, final_scale=1e-3, convergence_rate=1.0),
            covariance_schedule = AdaptiveExponentialSchedule(initial_scale=1.0,   final_scale=1e-3, convergence_rate=0.1),
        );
        params = optimal(prob.markov_chains[end]).param
        writeout(o, "Annealing", params)

        write(o, "Losses on General Strat Simulations: \n")
        total = 0
        for test_file in general_strat
                mymodel_test = build_model(test_file);
                tdata_test = mymodel_test.tdata
                model_test = mymodel_test.model
                loss = build_loss(model_test, tdata_test)
                loss = loss(params)
                write(o, "$(tdata.name): $(loss) \n")
                total += loss

                p = visualize_predictions(model_test, tdata_test, 1:90:length(tdata_test), params)
                PyPlot.savefig(directory*"Test/$(tdata_test.name).png")
        end
        write(o, "Mean loss on General Strat Simulations: $(total/length(general_strat)) \n")

        write(o, "Losses on 4 Day Suite Simulations: \n")
        total = 0
        for test_file in four_day_suite
                mymodel_test = build_model(test_file);
                tdata_test = mymodel_test.tdata
                model_test = mymodel_test.model
                loss = build_loss(model_test, tdata_test)
                loss = loss(params)
                write(o, "$(tdata.name): $(loss) \n")
                total += loss

                p = visualize_predictions(model_test, tdata_test, 1:90:length(tdata_test), params)
                PyPlot.savefig(directory*"Test/$(tdata_test.name).png")
        end
        write(o, "Mean loss on 4 Day Suite Simulations: $(total/length(four_day_suite)) \n")

        close(o)
end
