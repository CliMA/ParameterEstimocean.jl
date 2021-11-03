
cn = ConstrainedNormal(0.0,2.0,0.0,1e-5)

# Playing with ConstrainedNormal distributions
vals = [inverse_parameter_transform(cn, x) for x in rand(convert_prior(cn), 10000000)]

f = Figure()
axtop = Axis(f[1, 1])
CairoMakie.density!(axtop, vals)
CairoMakie.density!(axtop, rand(lognormal_with_mean_std(0.5e-5,0.25e-5), 10000000))
CairoMakie.xlims!(0,2e-5)
save("visualize_ConstrainedNormal.png", f)
display(f)

CairoMakie.density!(axright, ensemble[:, 2], direction = :y)

vlines!(axmain, [convective_κz], color=:red)
vlines!(axtop, [convective_κz], color=:red)
hlines!(axmain, [background_κz], color=:red)
hlines!(axright, [background_κz], color=:red)
colsize!(f.layout, 1, Fixed(300))
colsize!(f.layout, 2, Fixed(200))
rowsize!(f.layout, 1, Fixed(200))
rowsize!(f.layout, 2, Fixed(300))
leg = Legend(f[1, 2], scatters, ["Initial ensemble", "Iteration 1", "Iteration 2", "Iteration 10"], position = :lb)
hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)
save("distributions_makie.png", f)