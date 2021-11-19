# # Intro to observations
#
# This example demonstrates the features of `OneDimensionalTimeSeries`
# when constructed from  "synthetic observations" generated by an Oceananigans `Simulation`.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add OceanTurbulenceParameterEstimation, Oceananigans, CairoMakie"
# ```

# First we load few things

using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using CairoMakie

# # Generating synthetic observations
#
# We define a utility function for constructing synthetic observations,

default_closure = ConvectiveAdjustmentVerticalDiffusivity(; convective_κz = 1.0,
                                                            convective_νz = 0.9,
                                                            background_κz = 1e-4,
                                                            background_νz = 1e-5)

function generate_synthetic_observations(name = "convective_adjustment";
                                         Nz = 32,
                                         Lz = 64,
                                         Qᵇ = +1e-8,
                                         Qᵘ = -1e-5,
                                         Δt = 10.0,
                                         f₀ = 1e-4,
                                         N² = 1e-6,
                                         closure = default_closure)

    data_path = name * ".jld2"

    if isfile(data_path)
        return data_path
    end
    
    grid = RectilinearGrid(size=32, z=(-64, 0), topology=(Flat, Flat, Bounded))
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (; u=u_bcs, b=b_bcs),
                                        coriolis = FPlane(f=f₀),
                                        closure = closure)
                                        
    set!(model, b = (x, y, z) -> N² * z)
    
    simulation = Simulation(model; Δt, stop_time=12hours)

    function init_with_parameters(file, model)
        file["parameters"] = (; Qᵇ, Qᵘ, Δt)
        file["serialized/closure"] = closure
    end
    
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = TimeInterval(4hour),
                                                          prefix = name,
                                                          array_type = Array{Float64},
                                                          field_slicer = nothing,
                                                          init = init_with_parameters,
                                                          force = true)
    
    run!(simulation)

    return data_path
end

# and invoke it:

data_path = generate_synthetic_observations()

# # Specifying observations
#
# When synthetic observations are constructed from simulation data, we
# can select
#
# * The fields to include via `field_names`
#
# * Which data in the time-series to include via the `times` keyword.
#   This can be used to change the initial condition for a calibration run.
#
# For example, to build observations with a single field we write,

single_field_observations = OneDimensionalTimeSeries(data_path, field_names=:b, normalize=ZScore)

# To build observations with two fields we write

two_field_observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :b), normalize=ZScore)

# And to build observations with specified times we write

times = single_field_observations.times[2:end]
specified_times_observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :b), normalize=ZScore, times=times)

# Notice that in the last case, `specified_times_observations.times` is missing `0.0`.

# # Visualizing observations

# For this we include the initial condition and ``v`` velocity component,

observations = OneDimensionalTimeSeries(data_path, field_names=(:u, :v, :b), normalize=ZScore)

fig = Figure()

ax_b = Axis(fig[1, 1], xlabel = "Buoyancy [m s⁻²]")
ax_u = Axis(fig[1, 2], xlabel = "Velocities [m s⁻¹]")

z = znodes(Center, observations.grid)

colorcycle = [:black, :red, :blue, :orange]

for i = 1:length(observations.times)
    b = observations.field_time_serieses.b[i]
    u = observations.field_time_serieses.u[i]
    v = observations.field_time_serieses.v[i]
    t = observations.times[i]

    label = "t = " * prettytime(t)
    u_label = i == 1 ? "u, " * label : label
    v_label = i == 1 ? "v, " * label : label

    lines!(ax_b, interior(b)[1, 1, :], z; label, color=colorcycle[i])
    lines!(ax_u, interior(u)[1, 1, :], z; linestyle=:solid, color=colorcycle[i], label=u_label)
    lines!(ax_u, interior(v)[1, 1, :], z; linestyle=:dash, color=colorcycle[i], label=v_label)
end

axislegend(ax_b, position=:rb)
axislegend(ax_u, position=:lb, merge=true)

save("intro_to_observations.svg", fig)

# ![](intro_to_observations.svg)

# Hint: if using a REPL or notebook, try
# `using Pkg; Pkg.add("ElectronDisplay"); using ElectronDisplay; display(fig)`
# To see the figure in a window.

