using OceanTurbulenceParameterEstimation, Plots

# convert from offset arrays to normal arrays
tdata = OneDimensionalTimeSeries(files[1]*"/instantaneous_statistics.jld2")

# tdata = OneDimensionalTimeSeries("/Users/adelinehillier/.julia/dev/Data/WENO.jld2")
temp = tdata.T
Nz = tdata.grid.N

files = ["/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu0.0e+00_Qb1.0e-07_f1.0e-04_Nh256_Nz128_free_convection",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu0.0e+00_Qb8.0e-08_f1.0e-04_Nh256_Nz128_free_convection",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu1.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu1.0e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu4.0e-04_Qb3.0e-09_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu5.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu2.0e-04_Qb6.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu5.0e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu7.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind"
    ]

files = ["/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu1.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu0.0e+00_Qb7.0e-08_f1.0e-04_Nh256_Nz128_free_convection",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu8.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu7.0e-04_Qb3.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr96_Qu3.0e-04_Qb7.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling",
    ]

files = ["/Users/adelinehillier/.julia/dev/8DayLinearStrat/general_strat_4",
    "/Users/adelinehillier/.julia/dev/8DayLinearStrat/general_strat_8",
    "/Users/adelinehillier/.julia/dev/8DayLinearStrat/general_strat_16",
    "/Users/adelinehillier/.julia/dev/8DayLinearStrat/general_strat_32",
    ]

# files = ["/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr144_Qu0.0e+00_Qb6.0e-08_f1.0e-04_Nh256_Nz128_free_convection",
    # "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr144_Qu2.0e-04_Qb6.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling",
files = ["/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr144_Qu6.0e-05_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr144_Qu6.0e-04_Qb2.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling",
    # "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr144_Qu0.0e+00_Qb5.0e-08_f1.0e-04_Nh256_Nz128_free_convection",
    # "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr144_Qu2.0e-04_Qb5.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_cooling",
    "/Users/adelinehillier/.julia/dev/inst_to_transfer/three_layer_constant_fluxes_linear_hr144_Qu7.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind",
    ]

#!/bin/sh
# alias julia5='/home/ahillier/julia-1.5.2/bin/julia'
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/linear_stratification_constant_fluxes.jl --hours 192 --size 256 256 256 --extent 100 100 100 --momentum-flux 0.0 --buoyancy-flux 4.776e-8 --buoyancy-gradient 4.905e-6 --name general_strat_4 --animation
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/linear_stratification_constant_fluxes.jl --hours 192 --size 256 256 256 --extent 100 100 100 --momentum-flux 0.0 --buoyancy-flux 4.776e-8 --buoyancy-gradient 9.81e-6 --name general_strat_8 --animation
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/linear_stratification_constant_fluxes.jl --hours 192 --size 256 256 256 --extent 100 100 100 --momentum-flux 0.0 --buoyancy-flux 4.776e-8 --buoyancy-gradient 1.962e-5 --name general_strat_16 --animation
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/linear_stratification_constant_fluxes.jl --hours 192 --size 256 256 256 --extent 100 100 100 --momentum-flux 0.0 --buoyancy-flux 4.776e-8 --buoyancy-gradient 3.924e-5 --name general_strat_32 --animation
# CUDA_VISIBLE_DEVICES=3 julia5 --project examples/linear_stratification_constant_fluxes.jl --hours 5 --size 256 256 256 --extent 100 100 100 --momentum-flux 0.0 --buoyancy-flux 4.776e-8 --buoyancy-gradient 1.962e-5 --name general_strat_16 --animation

# files = ["Qu0.0e+00_Qb8.0e-08_free_convection", # good!
#          "Qu1.0e-04_Qb0.0e+00_strong_wind_no_rotation", # good!
#          "Qu2.0e-04_Qb6.0e-08_weak_wind_strong_cooling",
#          "Qu5.0e-04_Qb2.0e-08_strong_wind_weak_cooling",
#          "Qu7.0e-04_Qb0.0e+00_strong_wind"]

# strong wind = Qu7.0e-04
# strong cooling = Qb8.0e-08
# 2 is good!
# 3's good!
# 4 too shallow by 3/4
# 5 too shallow by 3/4
# 6 too shallow by 3/4

# 7e-4 strong wind
temperature(file) = [parent(x.data[1:Nz]) for x in OneDimensionalTimeSeries(file).T]
# uvelocity(file) = [parent(x.data[1:Nz]) for x in OneDimensionalTimeSeries(file).U]
# vvelocity(file) = [parent(x.data[1:Nz]) for x in OneDimensionalTimeSeries(file).V]
#
# get_mean_std(f) = mean([mean(std.(f(file*"/instantaneous_statistics.jld2"))) for file in files])
# get_mean_std(temperature)
# get_mean_std(uvelocity)
# get_mean_std(vvelocity)

number = 1
T = temperature(files[number]*"/instantaneous_statistics.jld2")
Nt = length(T)
# xlims = (minimum(minimum(T)), maximum(maximum(T)))
Plots.plot(T[Nt], collect(1:Nz), linewidth=2, la=0.5)

Plots.plot();
anim = @animate for n in 1:length(T)
    fig = Plots.plot(xlim=(19.6,20), legend=:bottom, size=(400,400), xlabel="Temperature (C)", ylabel="Depth (m)")
    Plots.plot!(fig, T[n], collect(1:Nz), linewidth=2, la=0.5, palette=:Set1_3)
end
gif(anim, pwd() * "/hello.gif", fps=40)
