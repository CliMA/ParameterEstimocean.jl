using Test
using DataDeps
using OceanTurbulenceParameterEstimation

using Statistics

@testset "Transformations tests" begin
    data_path = datadep"two_day_suite_2m/free_convection_instantaneous_statistics.jld2";
    field_names = :u

    raw_observations = SyntheticObservations(data_path; field_names = field_names)

    Nz = raw_observations.grid.Nz
    times = raw_observations.times

    time_indices = 101
    z_indices = [collect(20:2:Int(Nz/2)); Nz-4]

    time_transformation = time_indices
    space_transformation = SpaceIndices(y=:, z=z_indices)

    transformation = Transformation(space = space_transformation,
                                    time = time_transformation,
                                    normalization = nothing)

    observations₁ = SyntheticObservations(data_path; field_names = field_names, transformation)

    map = ConcatenatedOutputMap()

    sliced_raw_observations = raw_observations.field_time_serieses.u[:, :, z_indices, time_indices][1, 1, :, :]
    m, n = size(sliced_raw_observations)
    sliced_raw_observations = reshape(sliced_raw_observations, (m * n, 1))
    
    @test parent(observation_map(map, observations₁)) ≈ parent(sliced_raw_observations)

    time_indices = [100, 200]
    z_indices = 20:40

    time_transformation = time_indices
    space_transformation = SpaceIndices(y=:, z=z_indices)

    transformation = Transformation(space = space_transformation,
                                    time = time_transformation,
                                    normalization = ZScore())

    observations₂ = SyntheticObservations(data_path; field_names = field_names, transformation)

    sliced_raw_observations = raw_observations.field_time_serieses.u[:, :, z_indices, time_indices][1, 1, :, :]
    m, n = size(sliced_raw_observations)
    sliced_raw_observations = reshape(sliced_raw_observations, (m * n, 1))

    # normalize
    μ, σ =  mean(sliced_raw_observations), sqrt(cov(sliced_raw_observations; corrected=false))
    @. sliced_raw_observations = (sliced_raw_observations - μ) / σ

    @test parent(observation_map(map, observations₂)) ≈ parent(sliced_raw_observations)
end
