using OrderedCollections
using Oceananigans.OutputReaders: FieldTimeSeries

#####
##### The LESbrary (so to speak)
#####

include("./legacy_data_time_serieses.jl")

function get_times(path)
   file = jldopen(path)
   file_iterations = parse.(Int, keys(file["timeseries/t"]))
   file_times = [file["timeseries/t/$i"] for i in file_iterations]
   return file_times
end

# https://engaging-web.mit.edu/~alir/lesbrary/6DaySuite/
function SyntheticObservationsBatch(suite; first_iteration = 1, stride = 1, last_iteration = nothing, normalize = ZScore, Nz)

   observations = Vector{SyntheticObservations}()

   for case in values(suite)
      path = case.filename
      times = get_times(path)
      last_iteration = isnothing(last_iteration) ? length(times) : last_iteration
      times = times[first_iteration:stride:last_iteration]
      field_names = case.fields

      field_time_serieses = legacy_data_field_time_serieses(path, field_names, times)

      observation = SyntheticObservations(path; field_names, normalize, times, field_time_serieses, (1, 1, Nz))
      push!(observations, observation)
   end

   return observations
end

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

