# # Intro to observations
#
# This example demonstrates the features of `SyntheticObservations`
# when constructed from  "synthetic observations" generated by an Oceananigans `Simulation`.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add ParameterEstimocean, Oceananigans, CairoMakie"
# ```

# First we load few things

using ParameterEstimocean
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

function generate_synthetic_observations(name = "convective_adjustment"; Nz = 32, Lz = 64,
                                         Qᵇ = +1e-8, Qᵘ = -1e-5, f₀ = 1e-4, N² = 1e-6,
                                         Δt = 10.0, stop_time = 12hours, overwrite=false,
                                         output_schedule = TimeInterval(stop_time/3),
                                         tracers = :b, closure = default_closure)

    data_path = name * ".jld2"
  
    if isfile(data_path) && !overwrite
        @warn("Using existing data at $data_path. " *
              "Please delete this file if you wish to generate new data.")

        return data_path
    else
        overwrite_existing = true
    end
    
    grid = RectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ), bottom = GradientBoundaryCondition(N²))

    model = HydrostaticFreeSurfaceModel(; grid, tracers, closure,
                                          buoyancy = BuoyancyTracer(),
                                          boundary_conditions = (u=u_bcs, b=b_bcs),
                                          coriolis = FPlane(f=f₀))

    set!(model, b = (x, y, z) -> N² * z)
    simulation = Simulation(model; Δt, stop_time)
    init_with_parameters(file, model) = file["parameters"] = (; Qᵇ, Qᵘ, Δt, N², tracers=keys(model.tracers))
    
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers);
                                                          schedule = output_schedule,
                                                          filename = name,
                                                          array_type = Array{Float64},
                                                          with_halos = true,
                                                          init = init_with_parameters,
                                                          overwrite_existing)

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

single_field_observations = SyntheticObservations(data_path, field_names=:b, transformation=ZScore())

# To build observations with two fields we write

two_field_observations = SyntheticObservations(data_path, field_names=(:u, :b), transformation=ZScore())

# And to build observations with specified times we write

times = single_field_observations.times[2:end]
specified_times_observations = SyntheticObservations(data_path, field_names=(:u, :b), transformation=ZScore(), times=times)

# Notice that in the last case, `specified_times_observations.times` is missing `0.0`.

# # Visualizing observations

# For this we include the initial condition and ``v`` velocity component,

observations = SyntheticObservations(data_path, field_names=(:u, :v, :b), transformation=ZScore())

fig = Figure()

ax_b = Axis(fig[1, 1], xlabel = "Buoyancy [m s⁻²]", ylabel = "z [m]")
ax_u = Axis(fig[1, 2], xlabel = "Velocities [m s⁻¹]", ylabel = "z [m]")

z = znodes(observations.grid, Center())

colorcycle = [:black, :red, :blue, :orange, :pink]

for i = 1:length(observations.times)
    b_ = observations.field_time_serieses.b[i]
    u_ = observations.field_time_serieses.u[i]
    v_ = observations.field_time_serieses.v[i]
    t_ = observations.times[i]

    label = "t = " * prettytime(t_)
    u_label = i == 1 ? "u, " * label : label
    v_label = i == 1 ? "v, " * label : label

    lines!(ax_b, 1e4 * interior(b_)[1, 1, :], z; label, color=colorcycle[i]) # convert units from m s⁻² to 10⁻⁴ m s⁻²
    lines!(ax_u, interior(u_)[1, 1, :], z; linestyle=:solid, color=colorcycle[i], label=u_label)
    lines!(ax_u, interior(v_)[1, 1, :], z; linestyle=:dash, color=colorcycle[i], label=v_label)
end

axislegend(ax_b, position=:rb)
axislegend(ax_u, position=:lb, merge=true)

save("intro_to_observations.svg", fig)

# ![](intro_to_observations.svg)

# Hint: if using a REPL or notebook, try
# `using Pkg; Pkg.add("ElectronDisplay"); using ElectronDisplay; display(fig)`
# To see the figure in a window.

