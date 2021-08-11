
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, CATKEVerticalDiffusivity
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize

using BenchmarkTools

Ex, Ey = (400, 20)
sz = ColumnEnsembleSize(Nz=128, ensemble=(Ex, Ey))
halo = ColumnEnsembleSize(Nz=sz.Nz)

grid = RegularRectilinearGrid(size=sz, halo=halo, z=(-128, 0), topology=(Flat, Flat, Bounded))

closure = [CATKEVerticalDiffusivity() for i=1:Ex, j=1:Ey]
                                      
Qᵇ = [+1e-8 for i=1:Ex, j=1:Ey]
Qᵘ = [-1e-4 for i=1:Ex, j=1:Ey]
Qᵛ = [0.0   for i=1:Ex, j=1:Ey]

u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))
v_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵛ))
b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))

model = HydrostaticFreeSurfaceModel(architecture = CPU(),
                                    grid = grid,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=1e-4),
                                    boundary_conditions = (b=b_bcs, u=u_bcs, v=v_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)


function one_step!(model)
    time_step!(model, 1e-9)
    return nothing
end

@info "Running one time-step..."
one_step!(model)
