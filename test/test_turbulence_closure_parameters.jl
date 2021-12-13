using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity, MixingLength, SurfaceTKEFlux
using Suppressor: @suppress

using OceanTurbulenceParameterEstimation.TurbulenceClosureParameters: closure_with_parameters, update_closure_ensemble_member!

const CAVD = ConvectiveAdjustmentVerticalDiffusivity
const CATKE = CATKEVerticalDiffusivity

@testset "TurbulenceClosureParameters tests" begin
    Nz = 16
    Hz = 1
    grid = RectilinearGrid(size=Nz, z=(-10, 10), topology=(Flat, Flat, Bounded), halo=1)
    
    @info "Testing closure_with_parameters on ConvectiveAdjustmentVerticalDiffusivity"
    closures = [CAVD(background_κz=1.0) CAVD(background_κz=1.1)
                CAVD(background_κz=1.2) CAVD(background_κz=1.3)
                closure_with_parameters(CAVD(), (background_κz=1.4,)) closure_with_parameters(CAVD(), (background_κz=1.5,))]

    @test closures[3, 1].background_κz == 1.4
    @test closures[3, 2].background_κz == 1.5

    closure_tuple = closure_with_parameters((CAVD(), CAVD()), (background_κz = 14.0,))

    for closure in closure_tuple
        @test closure.background_κz == 14.0
    end

    @info "Testing closure_with_parameters on CATKEVerticalDiffusivity"

    catke_closure = @suppress CATKE()
    new_Cᴷuʳ = 1e3
    mixing_length = MixingLength(Cᴷuʳ = new_Cᴷuʳ)
    new_catke_closure = closure_with_parameters(catke_closure, (mixing_length = mixing_length,))
    @test new_catke_closure.mixing_length.Cᴷuʳ == new_Cᴷuʳ

    new_background_κz, new_convective_νz = 100.0, 10.0

    ensemble_member_no = 2
    update_closure_ensemble_member!(closures, ensemble_member_no, (background_κz = new_background_κz, convective_νz = new_convective_νz))

    @info "Testing update_closure_ensemble_member! on ConvectiveAdjustmentVerticalDiffusivity"
    for j in 1:2
        @test closures[ensemble_member_no, j].background_κz == new_background_κz
        @test closures[ensemble_member_no, j].convective_νz == new_convective_νz
    end

    new_CᴷRiʷ = 100.0
    mixing_length = MixingLength(CᴷRiʷ = new_CᴷRiʷ)

    closures = @suppress [CATKE() CATKE()
                          CATKE() CATKE()
                          CATKE() CATKE()]

    ensemble_member_no = 2
    update_closure_ensemble_member!(closures, ensemble_member_no, (mixing_length = mixing_length,))

    @info "Testing update_closure_ensemble_member! on CATKEVerticalDiffusivity"
    for j in 1:2
        @test closures[ensemble_member_no, j].mixing_length.CᴷRiʷ == new_CᴷRiʷ
    end
end
