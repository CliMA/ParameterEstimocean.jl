import Oceananigans.Models.HydrostaticFreeSurfaceModels: validate_test, validate_halo, time_discretization

struct ColumnEnsembleSize{C<:Tuple{Int, Int}}
    ensemble :: C
    Nz :: Int
end

ColumnEnsembleSize(; Nz, ensemble=(0, 0)) = ColumnEnsembleSize(ensemble, Nz)

validate_size(TX, TY, TZ, e::ColumnEnsembleSize) = tuple(e.ensemble[1], e.ensemble[2], e.Nz)
validate_halo(TX, TY, TZ, e::ColumnEnsembleSize) = tuple(e.ensemble[1], e.ensemble[2], e.Nz)

time_discretization(::AbstractArray{<:AbstractTurbulenceClosure{TD}}) where TD = TD()
