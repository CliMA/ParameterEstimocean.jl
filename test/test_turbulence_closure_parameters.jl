using Test
using OceanTurbulenceParameterEstimation
using Oceananigans
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

using OceanTurbulenceParameterEstimation.TurbulenceClosureParameters: closure_with_parameters, update_closure_ensemble_member!

const CAVD = ConvectiveAdjustmentVerticalDiffusivity

@testset "TurbulenceClosureParameters module" begin
    Nz = 16
    Hz = 1
    grid = RectilinearGrid(size=Nz, z=(-10, 10), topology=(Flat, Flat, Bounded), halo=1)
    
    closures = [CAVD(background_κz=1.0) CAVD(background_κz=1.1)
                CAVD(background_κz=1.2) CAVD(background_κz=1.3)
                closure_with_parameters(CAVD(), (background_κz=1.4,)) closure_with_parameters(CAVD(), (background_κz=1.5,))]

    @info "Testing closure_with_parameters"
    @test closures[3, 1].background_κz == 1.4
    @test closures[3, 2].background_κz == 1.5

    closure_tuple = closure_with_parameters((CAVD(), CAVD()), (background_κz = 14.0,))

    for closure in closure_tuple
        @test closure.background_κz == 14.0
    end

    new_background_κz, new_convective_νz = 100.0, 10.0

    ensemble_member_no = 2
    update_closure_ensemble_member!(closures, ensemble_member_no, (background_κz = new_background_κz, convective_νz = new_convective_νz))

    @info "Testing update_closure_ensemble_member!"
    for j in 1:2
        @test closures[ensemble_member_no, j].background_κz == new_background_κz
        @test closures[ensemble_member_no, j].convective_νz == new_convective_νz
    end
end
