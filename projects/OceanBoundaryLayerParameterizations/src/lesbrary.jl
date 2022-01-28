using DataDeps
using Statistics

fields_by_case = Dict(
   "free_convection" => (:b, :e),
   "weak_wind_strong_cooling" => (:b, :u, :v, :e),
   "strong_wind_weak_cooling" => (:b, :u, :v, :e),
   "strong_wind" => (:b, :u, :v, :e),
   "strong_wind_no_rotation" => (:b, :u, :e)
)

function SyntheticObservationsBatch(path_fn, normalization, times, Nz)

   observations = Vector{SyntheticObservations}()

   for (case, field_names) in zip(keys(fields_by_case), values(fields_by_case))

      data_path = @datadep_str path_fn(case)
      SyntheticObservations(data_path; field_names, normalization, times, regrid_size=(1, 1, Nz))

      push!(observations, observation)
   end

   return observations
end

two_day_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
four_day_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
six_day_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"

normalization = (b = ZScore(),
                 u = ZScore(),
                 v = ZScore(),
                 e = RescaledZScore(0.01)) 

TwoDaySuite(; normalization = normalization, times=[2hours, 12hours, 1days, 36hours, 2days], Nz=64) = SyntheticObservationsBatch(two_day_suite_path, normalization, times, Nz)
FourDaySuite(; normalization = normalization, times=[2hours, 1days, 2days, 3days, 4days], Nz=64) = SyntheticObservationsBatch(four_day_suite_path, normalization, times, Nz)
SixDaySuite(; normalization = normalization, times=[2hours, 1.5days, 3days, 4.5days, 6days], Nz=64) = SyntheticObservationsBatch(six_day_suite_path, normalization, times, Nz)

function lesbrary_ensemble_simulation(observations; ensemble_size = 30,
                                             architecture = CPU(),
                                             closure = ConvectiveAdjustmentVerticalDiffusivity(),
                                             Δt = 10.0
                                    )

    simulation = ensemble_column_model_simulation(observations;
                                                  Nensemble = ensemble_size,
                                                  architecture = architecture,
                                                  tracers = (:b, :e),
                                                  closure = closure)

    simulation.Δt = Δt

    Qᵘ = simulation.model.velocities.u.boundary_conditions.top.condition
    Qᵇ = simulation.model.tracers.b.boundary_conditions.top.condition
    N² = simulation.model.tracers.b.boundary_conditions.bottom.condition

    for (case, obs) in enumerate(observations)
        view(Qᵘ, case, :) .= obs.metadata.parameters.momentum_flux
        view(Qᵇ, case, :) .= obs.metadata.parameters.buoyancy_flux
        view(N², case, :) .= obs.metadata.parameters.N²_deep
    end

    return simulation
end

function estimate_η_covariance(output_map, observations)
    obs_maps = hcat([observation_map(output_map, obs) for obs in observations]...)
    return cov(transpose(obs_maps))
end 