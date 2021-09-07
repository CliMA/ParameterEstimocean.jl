using OceanTurbulenceParameterEstimation, Plots
using Flux: mse

# convert from offset arrays to normal arrays
tdata = TruthData("/Users/adelinehillier/.julia/dev/Data/WENO.jld2")
temp = tdata.T
Nz = tdata.grid.N
temperature(file) = [parent(x.data[1:Nz]) for x in TruthData(file).T]

WENO = temperature("/Users/adelinehillier/.julia/dev/Data/WENO.jld2")
AMD = temperature("/Users/adelinehillier/.julia/dev/Data/AMD.jld2")
WENO_AMD = temperature("/Users/adelinehillier/.julia/dev/Data/WENO_AMD.jld2")

mean(mse.(WENO, WENO_AMD))
mean(mse.(WENO, AMD))
mean(mse.(AMD, WENO_AMD))
a = [WENO, AMD, WENO_AMD]
legend_labels = ["WENO + Isotropitdiffusivity", "SecondOrder + AMD", "WENO + AMD"]

# Plots.plot(WENO[289], collect(1:Nz))
Plots.plot();
anim = @animate for n in 1:length(WENO)
    fig = Plots.plot(xlim=(19.8,20), legend=:bottom, size=(400,400), xlabel="Temperature (C)", ylabel="Depth (m)")
    for i in 1:length(a)
        Plots.plot!(fig, a[i][n], collect(1:Nz), label=legend_labels[i], linewidth=2, la=0.5, palette=:Set1_3)
    end
end
gif(anim, pwd() * "/compare_WENO_AMD.gif", fps=20)
