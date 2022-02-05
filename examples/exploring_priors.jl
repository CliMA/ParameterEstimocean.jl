# # Exploring priors
#
# This example explores specifying priors using OceanTurbulenceParameterEstimation. 
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, Distributions, CairoMakie"
# ```

# First we load few things

using OceanTurbulenceParameterEstimation
using CairoMakie
using Distributions

# `OceanTurbulenceParameterEstimation` supports three types of prior
# distributions. The normal prior distribution,

normal_prior = Normal(0.1, 0.1)

# the log-normal prior,

lognormal_prior = lognormal(mean=0.1, std=0.1)

# and the "scaled", logit-normal prior,

logitnormal_prior = ScaledLogitNormal(bounds=(0, 0.2))

# Note that `lognormal` is a constructor for `Lognormal(μ, σ)` that calculates
# the parameters `μ` and `σ` given a target distribution `mean` and 
# standard deviation `std`.
#
# Sampling the distributions reveals their properties:

samples = 10^6
fig = Figure()
ax = Axis(fig[1, 1:9], xlabel="Random samples of priors", ylabel="Density")

density!(ax, rand(normal_prior, samples), label="Normal")
density!(ax, rand(lognormal_prior, samples), label="Log-normal")
density!(ax, rand(logitnormal_prior, samples), label="Scaled logit-normal")

axislegend(ax)
xlims!(ax, -0.5, 1.0)

save("prior_flavors.svg", fig); nothing #hide 
# ![](prior_flavors.svg)

# We note three important features:
#
# 1. `Normal` samples can be negative.
# 2. `Lognormal` samples cannot be negative, but can have large positive values.
# 3. `ScaledLogitNormal` samples are _bounded_.
#
# Boundedness is a very useful property of `ScaledLogitNormal`, so we
# explore specifying `ScaledLogitNormal` priors in more detail.
#
# @doc ScaledLogitNormal

# We can specify `ScaledLogitNormal` by supplying the standard-deviation:

narrow = ScaledLogitNormal(σ=0.1)
default = ScaledLogitNormal(σ=1)
weird = ScaledLogitNormal(σ=4)

fig = Figure()
ax = Axis(fig[1, 1:9], xlabel="Random samples of priors", ylabel="Density")

density!(ax, rand(narrow, samples), label="narrow")
density!(ax, rand(default, samples), label="default")
density!(ax, rand(weird, samples), label="weird")

axislegend(ax)
xlims!(ax, -0.1, 1.1)

save("logit_normal_widths.svg", fig); nothing #hide 
# ![](logit_normal_widths.svg)

# Note we can get some weird shapes.
#
# Another way to build `ScaledLogitNormal` prior is to specify
# `interval` and `mass`. This allows us to shift the center of
# mass relative to the bounds (here we use the default `bounds=(0, 1)`.)

left = ScaledLogitNormal(interval=(0.2, 0.3), mass=0.9)
right = ScaledLogitNormal(interval=(0.7, 0.8), mass=0.9)
centered = ScaledLogitNormal(interval=(0.2, 0.8), mass=0.6)

fig = Figure()
ax = Axis(fig[1, 1:9], xlabel="Random samples of priors", ylabel="Density")

density!(ax, rand(left, samples), label="left")
density!(ax, rand(centered, samples), label="centered")
density!(ax, rand(right, samples), label="right")

axislegend(ax)
xlims!(ax, -0.1, 1.1)

save("logit_normal_intervals.svg", fig); nothing #hide 
# ![](logit_normal_intervals.svg)
#
