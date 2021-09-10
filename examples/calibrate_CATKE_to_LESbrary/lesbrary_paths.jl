#####
##### The LESbrary (so to speak)
#####

# mutable struct CaseMetaData{F, W, C, T}
#     filename :: F
#     with_wind_stress :: W
#     with_coriolis :: C
#     targets :: T
# end

# function CaseMetaData(; filename, with_wind_stress=true, with_coriolis=true, first=nothing, last=nothing, targets)
#    #targets = ...
#    metadata = CaseMetaData(filename, with_wind_stress, with_coriolis, targets)
#    return metadata
# end

# function stride_targets(metadata; stride=1)
#    first = metadata.targets[1]
#    last = metadata.targets[end]
#    new_targets = first:stride:last
#    return CaseMetaData(metadata.filename, metadata.with_wind_stress, metadata.with_coriolis, new_targets)
# end

# https://engaging-web.mit.edu/~alir/lesbrary/4DaySuite/
function FourDaySuite(directory)
   return OrderedDict(
                    "4d_free_convection" => (
                        filename = joinpath(directory, "free_convection/instantaneous_statistics.jld2"),
                        stressed = false,
                        rotating = true,
                           first = 13, # cut out the first 2 hours
                            last = nothing),

                    "4d_strong_wind" => (
                        filename = joinpath(directory, "strong_wind/instantaneous_statistics.jld2"),
                        stressed = true,
                        rotating = true,
                           first = 13,
                            last = nothing),

                    "4d_strong_wind_no_rotation" => (
                        filename = joinpath(directory, "strong_wind_no_rotation/instantaneous_statistics.jld2"),
                        stressed = true,
                        rotating = false,
                           first = 13,
                            last = nothing),

                     # This file is currently corrupted (on tartarus? on engaging?) and may need to be regenerated     
                     # "4d_strong_wind_weak_cooling" => (
                     #    filename = joinpath(directory, "strong_wind_weak_cooling/instantaneous_statistics.jld2"),
                     #    stressed = true,
                     #    rotating = true,
                     #       first = 13,
                     #        last = nothing),

                     "4d_weak_wind_strong_cooling" => (
                        filename = joinpath(directory, "weak_wind_strong_cooling/instantaneous_statistics.jld2"),
                        stressed = true,
                        rotating = true,
                           first = 13,
                            last = nothing),
                 )
end

# https://engaging-web.mit.edu/~alir/lesbrary/2DaySuite/
function TwoDaySuite(directory)
      return OrderedDict(
                   "2d_free_convection" => (
                       filename = joinpath(directory, "free_convection/instantaneous_statistics.jld2"),
                       stressed = false,
                       rotating = true,
                          first = 13, # cut out the first 2 hours
                           last = nothing),

                   "2d_strong_wind" => (
                       filename = joinpath(directory, "strong_wind/instantaneous_statistics.jld2"),
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),

                   "2d_strong_wind_no_rotation" => (
                       filename = joinpath(directory, "strong_wind_no_rotation/instantaneous_statistics.jld2"),
                       stressed = true,
                       rotating = false,
                          first = 13,
                           last = nothing),

                    "2d_strong_wind_weak_cooling" => (
                       filename = joinpath(directory, "strong_wind_weak_cooling/instantaneous_statistics.jld2"),
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),

                    "2d_weak_wind_strong_cooling" => (
                       filename = joinpath(directory, "weak_wind_strong_cooling/instantaneous_statistics.jld2"),
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),
                )
end

# https://engaging-web.mit.edu/~alir/lesbrary/6DaySuite/
function SixDaySuite(directory)
   return OrderedDict(
                 "6d_free_convection" => (
                     filename = joinpath(directory, "free_convection/instantaneous_statistics.jld2"),
                     stressed = false,
                     rotating = true,
                        first = 13, # cut out the first 2 hours
                         last = nothing),

                 "6d_strong_wind" => (
                     filename = joinpath(directory, "strong_wind/instantaneous_statistics.jld2"),
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),

                 "6d_strong_wind_no_rotation" => (
                     filename = joinpath(directory, "strong_wind_no_rotation/instantaneous_statistics.jld2"),
                     stressed = true,
                     rotating = false,
                        first = 13,
                         last = nothing),

                  "6d_strong_wind_weak_cooling" => (
                     filename = joinpath(directory, "strong_wind_weak_cooling/instantaneous_statistics.jld2"),
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),

                  "6d_weak_wind_strong_cooling" => (
                     filename = joinpath(directory, "weak_wind_strong_cooling/instantaneous_statistics.jld2"),
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),
              )
end
              
function GeneralStrat(directory)
   return OrderedDict(
                   "general_strat_4" => (
                       filename = joinpath(directory, "general_strat_4/instantaneous_statistics.jld2"),
                       stressed = false,
                       rotating = true,
                          first = 37, # cut out the first 6 hours
                           last = 288), # 2 days -- mixed layer depth reaches about 75 meters

                   "general_strat_8" => (
                       filename = joinpath(directory, "general_strat_8/instantaneous_statistics.jld2"),
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = 648), # 4 days -- mixed layer depth reaches about 75 meters

                   "general_strat_16" => (
                       filename = joinpath(directory, "general_strat_16/instantaneous_statistics.jld2"),
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = nothing),

                    "general_strat_32" => (
                       filename = joinpath(directory, "general_strat_32/instantaneous_statistics.jld2"),
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = nothing),
                )
end