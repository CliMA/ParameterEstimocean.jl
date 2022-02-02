using Test
using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using Oceananigans.TurbulenceClosures.CATKEVerticalDiffusivities: CATKEVerticalDiffusivity, MixingLength, SurfaceTKEFlux
using Suppressor: @suppress

using OceanTurbulenceParameterEstimation.Parameters: closure_with_parameters, update_closure_ensemble_member!

const CAVD = ConvectiveAdjustmentVerticalDiffusivity
const CATKE = CATKEVerticalDiffusivity

@testset "Parameters tests" begin
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
    new_catke_closure = closure_with_parameters(catke_closure, (; Cᴷuʳ = new_Cᴷuʳ))
    @test new_catke_closure.mixing_length.Cᴷuʳ == new_Cᴷuʳ

    @info "Testing update_closure_ensemble_member! on ConvectiveAdjustmentVerticalDiffusivity"
    new_background_κz, new_convective_νz = 100.0, 10.0
    ensemble_member_no = 2
    update_closure_ensemble_member!(closures, ensemble_member_no, (background_κz = new_background_κz, convective_νz = new_convective_νz))

    for j in 1:2
        @test closures[ensemble_member_no, j].background_κz == new_background_κz
        @test closures[ensemble_member_no, j].convective_νz == new_convective_νz
    end

    @info "Testing update_closure_ensemble_member! on CATKEVerticalDiffusivity"
    closures = @suppress [CATKE() CATKE()
                          CATKE() CATKE()
                          CATKE() CATKE()]

    new_CᴷRiʷ = 100.0
    ensemble_member_no = 2
    update_closure_ensemble_member!(closures, ensemble_member_no, (; CᴷRiʷ = new_CᴷRiʷ))

    for j in 1:2
        @test closures[ensemble_member_no, j].mixing_length.CᴷRiʷ == new_CᴷRiʷ
    end
end
