function normalized_counts(normal; bin_width=0.05, bin_range=(-10,10))
    bins = [bin_width*x for x=Int(10/bin_range[1]):Int(10/bin_range[2])]
    counts = Dict(x => 0 for x in bins)

    for x in normal
        bin = ceil(x/bin_width)*bin_width
        if bin in keys(counts)
            counts[bin] += 1
        end
    end

    area = sum(values(counts))*bin_width
    normalized = Dict(x => 0.0 for x in bins)
    for (bin, count) in counts
        normalized[bin] = count/area
    end

    return normalized
end

constrained_dist = Normal(7,1)
lognormal = log.([x for x in rand(constrained_dist,1000000) if x>0.0]);
lognormal_dist = fit(LogNormal, lognormal)
normal = exp.(rand(lognormal_dist, 1000000));


constrained_dist = Normal(0.001,0.1)
plot(constrained_dist, label="N(μ=0.001, σ=0.1)")

lognormal = [x for x in rand(constrained_dist,1000000) if x>0.0];
constrained_dist = fit(LogNormal, lognormal)
plot!(constrained_dist, label="lnN(μ=-2.93, σ=1.11)")

normal = log.([x for x in rand(constrained_dist,1000000) if x>0.0]);
normal_dist = fit(Normal, normal)

julia> mean(constrained_dist)
0.09840050052320308

julia> std(constrained_dist)
0.15290416640333576


plot!(normal_dist, label="N(-0.71,0.21)")

# lognormal = exp.(rand(normal_dist,1000000));
# normal_dist = fit(Normal, lognormal)



plot(constrained_dist, label = "normal N(7,1)")
plot!(normalized_counts(lognormal, bin_width=0.05), label = "log.(normal N(7,1))")
plot!(lognormal_dist, label = "lognormal N(0.657,0.078)")
plot!(normalized_counts(normal, bin_width=0.05), label="exp.(lognormal N(0.657,0.078))", legend=:topright)




constrained_dist = Normal(0.7,1)
lognormal = exp.([x for x in rand(constrained_dist,1000000) if x>0.0]);
lognormal_dist = fit(LogNormal, lognormal)
normal = log.(rand(lognormal_dist, 1000000));

plot(constrained_dist, label = "normal N(7,1)")
plot!(normalized_counts(lognormal, bin_width=0.05), label = "log.(normal N(7,1))")
plot!(lognormal_dist, label = "lognormal N(0.657,0.078)")
plot!(normalized_counts(normal, bin_width=0.05), label="exp.(lognormal N(0.657,0.078))", legend=:topright)





histogram(exp.rand(a, 100000))

constrained_prior = fit(LogNormal, log.(rand(Normal(5,1),10000)))


fit(LogNormal, rand(Normal(5,1),10000))
exp(5 + 0.5)

using StatsPlot
