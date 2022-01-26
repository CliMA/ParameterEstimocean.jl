using OrderedCollections
using Oceananigans.OutputReaders: FieldTimeSeries

#####
##### The LESbrary (so to speak)
#####

function get_times(path)
   file = jldopen(path)
   file_iterations = parse.(Int, keys(file["timeseries/t"]))
   file_times = [file["timeseries/t/$i"] for i in file_iterations]
   return file_times
end

fields_by_case = Dict(
   "free_convection" => (:b, :e),
   "weak_wind_strong_cooling" => (:b, :u, :v, :e),
   "strong_wind_weak_cooling" => (:b, :u, :v, :e),
   "strong_wind" => (:b, :u, :v, :e),
   "strong_wind_no_rotation" => (:b, :u, :e)
)

function SyntheticObservationsBatch(path_fn; normalize = ZScore, times=nothing, Nz=64)

   observations = Vector{SyntheticObservations}()

   for (case, field_names) in zip(keys(fields_by_case), values(fields_by_case))

      data_path = @datadep_str path_fn(case)
      SyntheticObservations(data_path; field_names, normalize, times, regrid_size=(1, 1, Nz))

      push!(observations, observation)
   end

   return observations
end

2s_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
4d_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"
6d_suite_path(case) = "two_day_suite_2m/$(case)_instantaneous_statistics.jld2"

2dSuite = SyntheticObservationsBatch(2s_suite_path; normalize = ZScore, times=[2hours, 12hours, 1days, 36hours, 2days], Nz=64)
4dSuite = SyntheticObservationsBatch(4d_suite_path; normalize = ZScore, times=[2hours, 1days, 2days, 3days, 4days], Nz=64)
6dSuite = SyntheticObservationsBatch(6d_suite_path; normalize = ZScore, times=[2hours, 1.5days, 3days, 4.5days, 6days], Nz=64)


# https://engaging-web.mit.edu/~alir/lesbrary/2DaySuite/
function TwoDaySuite(directory; first_iteration = 13, stride = 1, last_iteration = nothing, normalize = ZScore, Nz = 128)
   directory = joinpath(directory, "2DaySuite")
   suite = OrderedDict(
      "2d_free_convection" => (
         filename = joinpath(directory, "free_convection/instantaneous_statistics.jld2"),
         fields = (:b, :e)),
      "2d_strong_wind" => (
         filename = joinpath(directory, "strong_wind/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
      "2d_strong_wind_no_rotation" => (
         filename = joinpath(directory, "strong_wind_no_rotation/instantaneous_statistics.jld2"),
         fields = (:b, :u, :e)),
      "2d_strong_wind_weak_cooling" => (
         filename = joinpath(directory, "strong_wind_weak_cooling/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
      "2d_weak_wind_strong_cooling" => (
         filename = joinpath(directory, "weak_wind_strong_cooling/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
   )

   return SyntheticObservationsBatch(suite; first_iteration, stride, last_iteration, normalize, Nz)
end

# https://engaging-web.mit.edu/~alir/lesbrary/4DaySuite/
function FourDaySuite(directory; first_iteration = 13, stride = 1, last_iteration = nothing, normalize = ZScore, Nz = 128)
   directory = joinpath(directory, "4DaySuite")
   suite = OrderedDict(
      "4d_free_convection" => (
         filename = joinpath(directory, "free_convection/instantaneous_statistics.jld2"),
         fields = (:b, :e)),
      "4d_strong_wind" => (
         filename = joinpath(directory, "strong_wind/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
      "4d_strong_wind_no_rotation" => (
         filename = joinpath(directory, "strong_wind_no_rotation/instantaneous_statistics.jld2"),
         fields = (:b, :u, :e)),
      "4d_strong_wind_weak_cooling" => (
         filename = joinpath(directory, "strong_wind_weak_cooling/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
      "4d_weak_wind_strong_cooling" => (
         filename = joinpath(directory, "weak_wind_strong_cooling/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
   )

   return SyntheticObservationsBatch(suite; first_iteration, stride, last_iteration, normalize, Nz)
end

function SixDaySuite(directory; first_iteration = 13, stride = 1, last_iteration = nothing, normalize = ZScore, Nz = 128)
   directory = joinpath(directory, "6DaySuite")
   suite = OrderedDict(
      "6d_free_convection" => (
         filename = joinpath(directory, "free_convection/instantaneous_statistics.jld2"),
         fields = (:b, :e)),
      "6d_strong_wind" => (
         filename = joinpath(directory, "strong_wind/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
      "6d_strong_wind_no_rotation" => (
         filename = joinpath(directory, "strong_wind_no_rotation/instantaneous_statistics.jld2"),
         fields = (:b, :u, :e)),
      "6d_strong_wind_weak_cooling" => (
         filename = joinpath(directory, "strong_wind_weak_cooling/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
      "6d_weak_wind_strong_cooling" => (
         filename = joinpath(directory, "weak_wind_strong_cooling/instantaneous_statistics.jld2"),
         fields = (:b, :u, :v, :e)),
   )

   return SyntheticObservationsBatch(suite; first_iteration, stride, last_iteration, normalize, Nz)
end

function GeneralStrat(directory)
   directory = joinpath(directory, "6DaySuite")
   suite = OrderedDict(
      "general_strat_4" => (
         filename = joinpath(directory, "general_strat_4/instantaneous_statistics.jld2"),
         fields = (:b,),
         first = 37, # cut out the first 6 hours
         last = 288),
      "general_strat_8" => (
         filename = joinpath(directory, "general_strat_8/instantaneous_statistics.jld2"),
         fields = (:b,),
         first = 13,
         last = 648),
      "general_strat_16" => (
         filename = joinpath(directory, "general_strat_16/instantaneous_statistics.jld2"),
         fields = (:b,),
         first = 13,
         last = nothing),
      "general_strat_32" => (
         filename = joinpath(directory, "general_strat_32/instantaneous_statistics.jld2"),
         fields = (:b,),
         first = 13,
         last = nothing),
   )
end

