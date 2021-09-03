extent(grid::AbstractGrid) = (grid.Lx, grid.Ly, grid.Lz)

function get_grid_params(datapath::String)
    file = jldopen(datapath, "r")

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]
    Nz = file["grid/Nz"]

    Lx = file["grid/Lx"]
    Ly = file["grid/Ly"]
    Lz = file["grid/Lz"]

    close(file)
    return Nx, Ny, Nz, Lx, Ly, Lz
end

get_grid_params(grid::AbstractGrid) = tuple(size(grid)..., extent(grid)...)
