#
# Use this script to plot the evolution of the boundary layer depth
#

## Plotting mixed layer depth
using Plots, PyPlot
using Dao

using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.CATKEVerticalDiffusivityModel
using OceanTurbulenceParameterEstimation.ParameterEstimation
using OceanTurbulenceParameterEstimation.LossFunctions
using OceanTurbulenceParameterEstimation.ModelsAndData
using Oceananigans.Fields: interior

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

"""
Approximates the mixed layer depth for pure convection.
"""
function approximate_mixed_layer_depth(model_time_series, data::TruthData, targets)

        data.constants[:αg]
        Qᵇ = data.boundary_conditions.Qᵇ 
        N² = data.boundary_conditions.dbdz_bottom 

        Nz = data.grid.Nz
        z = data.grid.zC
        Lz = data.grid.Lz
        Nt = length(data.t)

        # For the LES solution
        h2_les = randn(Nt)
        for i in targets

            b = interior(data.b[i]) # remove halos

            Bz = δ([z...], [b...])
            mBz = maximum(Bz)
            tt = (2*N² + mBz)/3
            bools = Bz .> tt
            zA = (z[1:(Nz-1)] .+ z[2:Nz] ) ./ 2

            h2_les[i] = any(bools) ? -minimum(zA[bools]) : Lz
        end

        # For the model solution
        h2_model = randn(Nt)
        for i in 1:Nt-1
            B = interior(model_time_series.b[i])
            Bz = δ([z...], [B...])
            mBz = maximum(Bz)
            tt = (2*N² + mBz)/3
            bools = Bz .> tt
            zA = (z[1:(Nz-1)] + z[2:Nz] )./2
    
            h2_model[i] = any(bools) ? -minimum(zA[bools]) : Lz # mixed layer reached the bottom
        end

    return [h2_les, h2_model]
end

LESdata = GeneralStrat

ParametersToOptimize = TKEParametersRiDependent
RelevantParameters = ParametersToOptimize

params = Parameters(RelevantParameters = RelevantParameters,
               ParametersToOptimize = ParametersToOptimize)

function make_all_the_plots(params)

    directory = pwd() * "/Results/plotGeneralStrat_default_parameters/$(RelevantParameters)/"
    isdir(directory) || mkpath(directory)

    for (i, LEScase) in enumerate(values(LESdata))

        td = TruthData(LEScase.filename; grid_type=ColumnEnsembleGrid, Nz=32)

        relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0)

        # Single simulation
        case_loss, default_parameters = get_loss(LEScase, td, params, relative_weights; Δt=10.0, fields=relevant_fields(LEScase),
                                                parameter_specific_kwargs[params.RelevantParameters]...)

        targets = case_loss.loss.targets

        ℱ = model_time_series(default_parameters, case_loss.model, case_loss.data)
        h2_les, h2_model = approximate_mixed_layer_depth(ℱ, td, targets)

        days = td.t[targets] ./ 86400
        first = LEScase.first
        last = isnothing(LEScase.last) ? length(td.t) : LEScase.last

        Plots.plot(days, h2_les, label = "LES mixed layer depth", linewidth = 3 , color = :red, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box , title=td.name, ylabel = "Depth [meters]", xlabel = "days", legend = :topleft)
        p = plot!(days, h2_model, color = :purple, label = "TKE mixed layer depth", linewidth = 3 )
        Plots.savefig(p, directory*td.name*"_mixed_layer_depth.pdf")

        Plots.plot(days, h2_les, label = "LES mixed layer depth", linewidth = 3 , color = :red, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box , title=td.name, ylabel = "Depth [meters]", xlabel = "days", yscale = :log10, xscale = :log10, legend = :topleft)
        p = plot!(days, h2_model, color = :purple, label = "TKE mixed layer depth", linewidth = 3 )
        Plots.savefig(p, directory*td.name*"_mixed_layer_depth_log10.pdf")

        # p = visualize_realizations(md, td, 1:180:length(td), best_parameters)
        # PyPlot.savefig(directory*td.name*".png")
    end
end

make_all_the_plots(params)