using Test
using DataDeps
using ParameterEstimocean

using Statistics

@testset "Transformations tests" begin
    data_path = datadep"two_day_suite_2m/free_convection_instantaneous_statistics.jld2";
    field_names = :u

    raw_observations = SyntheticObservations(data_path; field_names = field_names)
    Nz = raw_observations.grid.Nz
    times = raw_observations.times

    map = ConcatenatedOutputMap()

    @info "  Test space and time slicing..."
    time_indices = 101
    z_indices = [collect(20:2:Int(Nz/2)); Nz-4]

    time_transformation = TimeIndices(t=time_indices)
    space_transformation = SpaceIndices(y=:, z=z_indices)

    transformation = Transformation(space = space_transformation,
                                    time = time_transformation,
                                    normalization = nothing)

    observations = SyntheticObservations(data_path; field_names = field_names, transformation)

    sliced_raw_observations = raw_observations.field_time_serieses.u[:, :, z_indices, time_indices][1, 1, :, :]
    ny, nz = size(sliced_raw_observations)
    sliced_raw_observations = reshape(sliced_raw_observations, (ny * nz, 1))
    
    @test parent(observation_map(map, observations)) ≈ parent(sliced_raw_observations)

    time_indices = [100, 200]
    z_indices = 20:40

    time_transformation = TimeIndices(t=time_indices)
    space_transformation = SpaceIndices(y=:, z=z_indices)

    transformation = Transformation(space = space_transformation,
                                    time = time_transformation,
                                    normalization = ZScore())

    observations = SyntheticObservations(data_path; field_names = field_names, transformation)

    sliced_raw_observations = raw_observations.field_time_serieses.u[:, :, z_indices, time_indices][1, 1, :, :]
    ny, nz = size(sliced_raw_observations)
    sliced_raw_observations = reshape(sliced_raw_observations, (ny * nz, 1))

    # normalize
    μ, σ =  mean(sliced_raw_observations), sqrt(cov(sliced_raw_observations; corrected=true))
    @. sliced_raw_observations = (sliced_raw_observations - μ) / σ

    @test parent(observation_map(map, observations)) ≈ parent(sliced_raw_observations)

    @info "  Test applying space and time weights..."

    time_scale_factor = 0.55
    time_weights = time_scale_factor * ones(size(times))

    depth_scale = 100
    z = raw_observations.grid.zᵃᵃᶜ[1:Nz]
    z_weights = @. exp(z / depth_scale)
    space_weights = reshape(z_weights, 1, 1, Nz)

    transformation = Transformation(space = space_weights,
                                    time = time_weights,
                                    normalization = nothing)

    observations = SyntheticObservations(data_path; field_names = field_names, transformation)

    weighted_raw_observations = raw_observations.field_time_serieses.u[1, 1, 1:Nz, :]
    @. weighted_raw_observations *= z_weights * time_scale_factor
    ny, nz = size(weighted_raw_observations)
    weighted_raw_observations = reshape(weighted_raw_observations, (ny * nz, 1))

    @test parent(observation_map(map, observations)) ≈ parent(weighted_raw_observations)
end
