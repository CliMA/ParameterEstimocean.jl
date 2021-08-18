#
# Use this script to plot the evolution of the boundary layer depth
#

## Plotting mixed layer depth
using Plots, PyPlot
using Dao

using OceanTurbulenceParameterEstimation
using OceanTurbulenceParameterEstimation.TKEMassFluxModel
using OceanTurbulenceParameterEstimation.ParameterEstimation

using OceanTurbulenceParameterEstimation: visualize_realizations, model_time_series

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
function approximate_mixed_layer_depth(model_time_series, data::TruthData; coarse_grain_data = false)

        data.constants[:αg]
        Qᵇ = 0.001962 * tdata.boundary_conditions.Qᶿ # les.α * les.g * les.top_T
        N² = 0.001962 * tdata.boundary_conditions.dθdz_bottom # les.α * les.g * dθdz_bottom
        Nt = length(tdata.t)

        if coarse_grain_data
            Nz = model_time_series.b[1].grid.Nz
            z = model_time_series.b[1].grid.zC
        else
            Nz = tdata.grid.Nz
            z = tdata.grid.zC
        end

        # For the LES solution
        h2_les = randn(Nt)
        for i in 1:Nt-20

            if coarse_grain_data
                b = CenterField(model_time_series.b[i].grid)
                set!(b, tdata.b[i])
                b = b.data[1:Nz]
            else
                b = tdata.b[i].data[1:Nz] # remove halos
            end

            Bz = δ([z...], [b...])
            mBz = maximum(Bz)
            tt = (2*N² + mBz)/3
            bools = Bz .> tt
            zA = (z[1:(Nz-1)] .+ z[2:Nz] ) ./ 2

            h2_les[i] = any(bools) ? -minimum(zA[bools]) : model_time_series.b[1].grid.Lz
        end

        @info Nt
        # For the model solution
        h2_model = randn(Nt)
        z = model_time_series.b[1].grid.zC
        Nz = model_time_series.b[1].grid.Nz
        for i in 1:Nt-20
            B = model_time_series.b[i].data
            B = B[1:Nz]
            Bz = δ([z...], [B...])
            mBz = maximum(Bz)
            tt = (2*N² + mBz)/3
            bools = Bz .> tt
            zA = (z[1:(Nz-1)] + z[2:Nz] )./2
    
            h2_model[i] = any(bools) ? -minimum(zA[bools]) : 
                                        model_time_series.b[1].grid.Lz # mixed layer reached the bottom
        end

    return [h2_les, h2_model]
end

LESdata = FourDaySuite

ParametersToOptimize = TKEParametersRiIndependentConvectiveAdjustment
RelevantParameters = TKEParametersRiIndependentConvectiveAdjustment

params = Parameters(RelevantParameters = RelevantParameters,
               ParametersToOptimize = ParametersToOptimize)

function make_all_the_plots(params)

    directory = pwd() * "/Results/plotFourDaySuite_default_parameters/$(RelevantParameters)/"
    isdir(directory) || mkpath(directory)

    for (i, LEScase) in enumerate(values(LESdata))

        td = TruthData(LEScase.filename; grid_type=ManyIndependentColumns, Nz=64)

        relative_weights = Dict(:b => 1.0, :u => 1.0, :v => 1.0, :e => 1.0)

        # Single simulation
        case_loss, default_parameters = get_loss(LEScase, td, params, relative_weights; Δt=10.0, 
                                                parameter_specific_kwargs[params.RelevantParameters]...)


        ℱ = model_time_series(default_parameters, case_loss)
        h2_les, h2_model = get_h2(ℱ, td; coarse_grain_data = false)

        days = @. td.t / 86400
        first = LEScase.first
        last = isnothing(LEScase.last) ? length(td.t) : LEScase.last

        toplot = LEScase.first:last
        Plots.plot(days[toplot], h2_les[toplot], label = "LES mixed layer depth", linewidth = 3 , color = :red, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box , title=td.name, ylabel = "Depth [meters]", xlabel = "days", legend = :topleft)
        p = plot!(days[toplot], h2_model[toplot], color = :purple, label = "TKE mixed layer depth", linewidth = 3 )
        Plots.savefig(p, directory*td.name*"_mixed_layer_depth.pdf")

        Plots.plot(days[toplot], h2_les[toplot], label = "LES mixed layer depth", linewidth = 3 , color = :red, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box , title=td.name, ylabel = "Depth [meters]", xlabel = "days", yscale = :log10, xscale = :log10, legend = :topleft)
        p = plot!(days[toplot], h2_model[toplot], color = :purple, label = "TKE mixed layer depth", linewidth = 3 )
        Plots.savefig(p, directory*td.name*"_mixed_layer_depth_log10.pdf")

        # p = visualize_realizations(md, td, 1:180:length(td), best_parameters)
        # PyPlot.savefig(directory*td.name*".png")
    end
end

make_all_the_plots(params)