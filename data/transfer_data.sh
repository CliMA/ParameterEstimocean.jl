#!/bin/bash

dir1="three_layer_constant_fluxes_linear_hr96_Qu0.0e+00_Qb7.0e-08_f1.0e-04_Nh256_Nz128_free_convection"
dir2="three_layer_constant_fluxes_linear_hr96_Qu1.0e-04_Qb0.0e+00_f0.0e+00_Nh256_Nz128_strong_wind_no_rotation"
dir3="three_layer_constant_fluxes_linear_hr96_Qu3.0e-04_Qb7.0e-08_f1.0e-04_Nh256_Nz128_weak_wind_strong_coolina"
dir4="three_layer_constant_fluxes_linear_hr96_Qu6.5e-04_Qb4.0e-08_f1.0e-04_Nh256_Nz128_strong_wind_weak_cooling"
dir5="three_layer_constant_fluxes_linear_hr96_Qu8.0e-04_Qb0.0e+00_f1.0e-04_Nh256_Nz128_strong_wind"

mkdir $dir1
scp tartarus:/home/greg/Projects/animations/4DaySuite/$dir1/instantaneous_statistics.jld2 $dir1/

mkdir $dir2
scp tartarus:/home/greg/Projects/animations/4DaySuite/$dir2/instantaneous_statistics.jld2 $dir2/

mkdir $dir3
scp tartarus:/home/greg/Projects/animations/4DaySuite/$dir3/instantaneous_statistics.jld2 $dir3/

mkdir $dir4
scp tartarus:/home/greg/Projects/animations/4DaySuite/$dir4/instantaneous_statistics.jld2 $dir4/

mkdir $dir5
scp tartarus:/home/greg/Projects/animations/4DaySuite/$dir5/instantaneous_statistics.jld2 $dir5/

