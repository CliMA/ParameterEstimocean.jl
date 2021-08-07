## Plotting mixed layer depth

using Statistics, Distributions, PyPlot, Plots
using OceanTurb, OceanTurbulenceParameterEstimation, Dao
using OceanTurbulenceParameterEstimation.TKEMassFluxOptimization
using OceanTurbulenceParameterEstimation.TKEMassFluxOptimization: ParameterizedModel
using Optim

prefix = "/Users/adelinehillier/.julia/dev/"
suffix = "instantaneous_statistics.jld2"
files = ["8DayLinearStrat/general_strat_4/",
         "8DayLinearStrat/general_strat_8/",
         "8DayLinearStrat/general_strat_16/",
         "8DayLinearStrat/general_strat_32/"
         ]
files = [prefix*x*suffix for x in files]

to_calibrate = files[4]

include("my_models.jl")
build_model = tke_free_convection_simple
# build_model = conv_adj_independent_diffusivites

mymodel = build_model(to_calibrate);
tdata = mymodel.tdata
model = mymodel.model
ParametersToOptimize = mymodel.ParametersToOptimize
default_parameters = mymodel.default_parameters
bounds = mymodel.bounds
loss_function, loss = build_loss(model, tdata)
loss(default_parameters)


directory = pwd() * "/TKECalibration2021Results/mixed_layer_depth_vs_time/$(RelevantParameters)/"
isdir(directory) || mkpath(directory)

@info "Running Iterative Simulated Annealing..."

variance = Array(ParametersToOptimize((0.1 * bound[2] for bound in bounds)...))
prob = anneal(loss, default_parameters, variance, BoundedNormalPerturbation, bounds;
             iterations = 10,
                samples = 1000,
     annealing_schedule = AdaptiveExponentialSchedule(initial_scale=100.0, final_scale=1e-3, convergence_rate=1.0),
    covariance_schedule = AdaptiveExponentialSchedule(initial_scale=1.0,   final_scale=1e-3, convergence_rate=0.1),
);
params = optimal(prob.markov_chains[end]).param

# les analysis
# derivative function
function δ(z, Φ)
    m = length(Φ)-1
    Φz = ones(m)
    for i in 1:m
        Φz[i] = (Φ[i+1]-Φ[i])/(z[i+1]-z[i])
    end
    return Φz
end

function get_h2(model_time_series, tdata; coarse_grain_data = true)

        # model.model.constants.g * model.model.constants.α

        Qᵇ = 0.001962 * tdata.boundary_conditions.Qᶿ # les.α * les.g * les.top_T
        N² = 0.001962 * tdata.boundary_conditions.dθdz_bottom # les.α * les.g * dθdz_bottom
        Nt = length(tdata.t)

        if coarse_grain_data
            Nz = model_time_series.T[1].grid.N
            z = model_time_series.T[1].grid.zc
        else
            Nz = tdata.grid.N
            z = tdata.grid.zc
        end

        # For the LES solution
        h2_les = randn(Nt)
        for i in 1:Nt

            if coarse_grain_data
                T = CellField(model_time_series.T[i].grid)
                set!(T, tdata.T[i])
                T = T.data[1:Nz]
            else
                T = tdata.T[i].data[1:Nz] # remove halos
            end

            B = 0.001962 * T
            Bz = δ([z...], [B...])
            mBz = maximum(Bz)
            tt = (2*N² + mBz)/3
            bools = Bz .> tt
            zA = (z[1:(end-1)] + z[2:end] )./2
            h2_les[i] = -minimum(zA[bools])
        end

        # For the model solution
        h2_model = randn(Nt)
        z = model_time_series.T[1].grid.zc
        Nz = model_time_series.T[1].grid.N
        for i in 1:Nt
            B = 0.001962 * model_time_series.T[i].data
            B = B[1:Nz]
            Bz = δ([z...], [B...])
            mBz = maximum(Bz)
            tt = (2*N² + mBz)/3
            bools = Bz .> tt
            zA = (z[1:(end-1)] + z[2:end] )./2

            if 1 in bools
                h2_model[i] = -minimum(zA[bools])
            else
                h2_model[i] = model_time_series.T[1].grid.H # mixed layer reached the bottom
            end
        end

    return [h2_les, h2_model]
end


for file in files
    mymodel = build_model(file);
    td = mymodel.tdata
    md = mymodel.model

    ℱ = model_time_series(params, md, td)
    h2_les, h2_model = get_h2(ℱ, td; coarse_grain_data = false)

    days = @. td.t / 86400
    toplot = 3:length(td.t)
    Plots.plot(days[toplot], h2_les[toplot], label = "LES mixed layer depth", linewidth = 3 , color = :red, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box , title=td.name, ylabel = "Depth [meters]", xlabel = "days", xlims = (0.1, 8.0) , legend = :topleft)
    p = plot!(days[toplot], h2_model[toplot], color = :purple, label = "TKE mixed layer depth", linewidth = 3 )
    Plots.savefig(p, directory*td.name*"_mixed_layer_depth.pdf")

    p = visualize_realizations(md, td, 1:180:length(td), params)
    PyPlot.savefig(directory*td.name*".png")
end
