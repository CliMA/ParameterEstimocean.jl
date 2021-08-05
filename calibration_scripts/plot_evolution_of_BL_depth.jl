## Plotting mixed layer depth
using TKECalibration2021
using Plots, PyPlot
using OceanTurbulenceParameterEstimation: visualize_realizations, model_time_series
using Dao

             LESdata = GeneralStrat
  RelevantParameters = TKEParametersConvectiveAdjustmentRiIndependent
ParametersToOptimize = TKEParametersConvectiveAdjustmentRiIndependent

dname = "calibrate_FourDaySuite_validate_GeneralStrat"
directory = pwd() * "/TKECalibration2021Results/compare_calibration_algorithms/$(dname)/$(RelevantParameters)/"
isdir(directory) || mkpath(directory)

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

# best_parameters = ParametersToOptimize([2.1638101987647502, 0.2172594537369187, 0.4522886369267623, 0.7534625713891345, 0.4477179760916435, 6.777679962252731, 1.2403584780163417, 1.9967245163343093])

for LEScase in values(LESdata)
    case_nll, _ = custom_tke_calibration(LEScase, RelevantParameters, ParametersToOptimize)

    td = case_nll.data
    md = case_nll.model

    ℱ = model_time_series(best_parameters, md, td)
    h2_les, h2_model = get_h2(ℱ, td; coarse_grain_data = false)

    days = @. td.t / 86400
    first = LEScase.first
    last = LEScase.last
    if LEScase.last == nothing
        last = length(td.t)
    end

    toplot = LEScase.first:last
    Plots.plot(days[toplot], h2_les[toplot], label = "LES mixed layer depth", linewidth = 3 , color = :red, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box , title=td.name, ylabel = "Depth [meters]", xlabel = "days", legend = :topleft)
    p = plot!(days[toplot], h2_model[toplot], color = :purple, label = "TKE mixed layer depth", linewidth = 3 )
    Plots.savefig(p, directory*td.name*"_mixed_layer_depth.pdf")

    Plots.plot(days[toplot], h2_les[toplot], label = "LES mixed layer depth", linewidth = 3 , color = :red, grid = true, gridstyle = :dash, gridalpha = 0.25, framestyle = :box , title=td.name, ylabel = "Depth [meters]", xlabel = "days", yscale = :log10, xscale = :log10, legend = :topleft)
    p = plot!(days[toplot], h2_model[toplot], color = :purple, label = "TKE mixed layer depth", linewidth = 3 )
    Plots.savefig(p, directory*td.name*"_mixed_layer_depth_log10.pdf")

    p = visualize_realizations(md, td, 1:180:length(td), best_parameters)
    PyPlot.savefig(directory*td.name*".png")
end
