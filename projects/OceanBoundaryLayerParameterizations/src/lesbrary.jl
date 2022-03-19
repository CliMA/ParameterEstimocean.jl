using DataDeps
using Statistics

fields_by_case = Dict(
   "free_convection" => (:b, :e),
   "weak_wind_strong_cooling" => (:b, :u, :v, :e),
   "strong_wind_weak_cooling" => (:b, :u, :v, :e),
   "strong_wind" => (:b, :u, :v, :e),
   "strong_wind_no_rotation" => (:b, :u, :e)
)

using OceanLearning.Transformations: Transformation

function SyntheticObservationsBatch(path_fn, transformation, times, Nz)

   observations = Vector{SyntheticObservations}()
   field_names = (:b, :u, :v, :e)

   for (case, forward_map_names) in zip(keys(fields_by_case), values(fields_by_case))

      data_path = @datadep_str path_fn(case)
      SyntheticObservations(data_path; transformation, times, field_names, forward_map_names, regrid=(1, 1, Nz))

      push!(observations, observation)
   end

   return observations
end

two_day_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
four_day_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
six_day_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"

transformation = (b = Transformation(normalization=ZScore()),
                  u = Transformation(normalization=ZScore()),
                  v = Transformation(normalization=ZScore()),
                  e = Transformation(normalization=RescaledZScore(1e-1)))

TwoDaySuite(; transformation, times=[2hours, 12hours, 1days, 36hours, 2days], Nz=64) = SyntheticObservationsBatch(two_day_suite_path, transformation, times, Nz)
FourDaySuite(; transformation, times=[2hours, 1days, 2days, 3days, 4days], Nz=64) = SyntheticObservationsBatch(four_day_suite_path, transformation, times, Nz)
SixDaySuite(; transformation, times=[2hours, 1.5days, 3days, 4.5days, 6days], Nz=64) = SyntheticObservationsBatch(six_day_suite_path, transformation, times, Nz)

function lesbrary_ensemble_simulation(observations; 
                                             Nensemble = 30,
                                             architecture = CPU(),
                                             closure = ConvectiveAdjustmentVerticalDiffusivity(),
                                             Δt = 10.0
                                    )

    simulation = ensemble_column_model_simulation(observations;
                                                  Nensemble,
                                                  architecture,
                                                  tracers = (:b, :e),
                                                  closure)

    simulation.Δt = Δt

    Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
    Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
    N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

    for (case, obs) in enumerate(observations)
        view(Qᵘ, :, case) .= obs.metadata.parameters.momentum_flux
        view(Qᵇ, :, case) .= obs.metadata.parameters.buoyancy_flux
        view(N², :, case) .= obs.metadata.parameters.N²_deep
    end

    return simulation
end

function estimate_η_covariance(output_map, observations)
    obs_maps = hcat([observation_map(output_map, obs) for obs in observations]...)
    return cov(transpose(obs_maps))
end