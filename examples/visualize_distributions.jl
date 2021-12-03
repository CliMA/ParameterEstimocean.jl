
###
### Visualize the prior densities
###

cn = ConstrainedNormal(0.0,2.0,0.0,1e-5)

# Sample from ConstrainedNormal 
vals = [inverse_parameter_transform(cn, x) for x in rand(convert_prior(cn), 10000000)]

f = Figure()
axtop = Axis(f[1, 1])
density!(axtop, vals)
# xlims!(0,2e-5)
save("visualize_ConstrainedNormal.png", f)
display(f)

density!(axright, ensemble[:, 2], direction = :y)

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