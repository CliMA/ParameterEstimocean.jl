###
### Grids
###

get_grid_params(g::Abstract) = [size(g); extent(g)]

"""
  ColumnEnsembleGrid(datapath; ensemble_size = (1,1), Nz = nothing)

Construct a 3D grid of type `Oceananigans.RegularRectilinearGrid` 
with independent columns, each with `Nz` grid points and
arranged horizontally according to `(Nx, Ny)`.
The grid is identical in extent to that of the LES simulation
stored in `datapath`. `Nz` defaults to the simulation resolution.
"""
function ColumnEnsembleGrid(datapath; size = (1,1,nothing))

      _, _, Nz_, _, _, Lz_ = get_grid_params(datapath)

      Nz = isnothing(size[3]) ? Nz_ : size[3]

      @assert Nz <=  Nz_ "Desired grid resolution exceeds the simulation resolution!"

      sz = ColumnEnsembleSize(Nz=Nz, ensemble=size[1:2])
      halo = ColumnEnsembleSize(Nz=Nz)

      grid = RegularRectilinearGrid(size=sz, halo=halo, z=(-Lz_, 0), topology=(Flat, Flat, Bounded))
end

# """
#   XYZGrid(N)

#   Construct a 3D grid of type `Oceananigans.RegularRectilinearGrid` 
#   with size `size` = (Nx, Ny, Nz) and the same extent as the that
#   of the LES simulation stored in `datapath`. 
#   `size` defaults to the simulation resolution.
# """
# function XYZGrid(datapath; size=nothing)

#     Nx, Ny, Nz, Lx, Ly, Lz = get_grid_params(datapath)

#     size = isnothing(size) ? (Nx, Ny, Nz) : size

#     @assert all( size .<= (Nx, Ny, Nz) ) "Desired grid resolution exceeds the simulation resolution!"

#     grid = RegularRectilinearGrid(size=(Nx, Ny, Nz), extent=(Lx, Ly, Lz), topology=(Periodic, Periodic, Bounded))
#     return grid
# end