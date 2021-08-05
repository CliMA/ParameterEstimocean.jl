###
### Grids
###

"""
  ZGrid(N)

  Construct a 1D column grid of type `Oceananigans.RegularRectilinearGrid` 
  with `N` grid points and the same extent as that of the LES simulation
  stored in `datapath`. `N` defaults to the simulation resolution.
"""
function ZGrid(datapath; size=nothing)

    _, _, Nz, _, _, Lz = get_grid_params(datapath)

    size = isnothing(size) ? Nz : size

    @assert size <=  Nz "Desired grid resolution exceeds the simulation resolution!"

    grid = RegularRectilinearGrid(size=size, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
    return grid
end

"""
  XYZGrid(N)

  Construct a 3D grid of type `Oceananigans.RegularRectilinearGrid` 
  with size `size` = (Nx, Ny, Nz) and the same extent as the that
  of the LES simulation stored in `datapath`. 
  `size` defaults to the simulation resolution.
"""
function XYZGrid(datapath; size=nothing)

    Nx, Ny, Nz, Lx, Ly, Lz = get_grid_params(datapath)

    size = isnothing(size) ? (Nx, Ny, Nz) : size

    @assert all( size .<= (Nx, Ny, Nz) ) "Desired grid resolution exceeds the simulation resolution!"

    grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))
    return grid
end