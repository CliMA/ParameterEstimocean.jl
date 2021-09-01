#####
##### The LESbrary (so to speak)
#####

# https://engaging-web.mit.edu/~alir/lesbrary/4DaySuite/

FourDaySuite_path = "/Users/gregorywagner/Projects/OceanTurbulenceParameterEstimation/data/"

FourDaySuite = OrderedDict(
                    "4d_free_convection" => (
                        filename = FourDaySuite_path*"free_convection/instantaneous_statistics.jld2",
                        stressed = false,
                        rotating = true,
                           first = 13, # cut out the first 2 hours
                            last = nothing),

                    "4d_strong_wind" => (
                        filename = FourDaySuite_path*"strong_wind/instantaneous_statistics.jld2",
                        stressed = true,
                        rotating = true,
                           first = 13,
                            last = nothing),

                    "4d_strong_wind_no_rotation" => (
                        filename = FourDaySuite_path*"strong_wind_no_rotation/instantaneous_statistics.jld2",
                        stressed = true,
                        rotating = false,
                           first = 13,
                            last = nothing),

                     "4d_strong_wind_weak_cooling" => (
                        filename = FourDaySuite_path*"strong_wind_weak_cooling/instantaneous_statistics.jld2",
                        stressed = true,
                        rotating = true,
                           first = 13,
                            last = nothing),

                     "4d_weak_wind_strong_cooling" => (
                        filename = FourDaySuite_path*"weak_wind_strong_cooling/instantaneous_statistics.jld2",
                        stressed = true,
                        rotating = true,
                           first = 13,
                            last = nothing),
                 )

# https://engaging-web.mit.edu/~alir/lesbrary/2DaySuite/
TwoDaySuite_path = "/Users/adelinehillier/.julia/dev/2DaySuite/"
TwoDaySuite = OrderedDict(
                   "2d_free_convection" => (
                       filename = TwoDaySuite_path*"free_convection/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 13, # cut out the first 2 hours
                           last = nothing),

                   "2d_strong_wind" => (
                       filename = TwoDaySuite_path*"strong_wind/instantaneous_statistics.jld2",
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),

                   "2d_strong_wind_no_rotation" => (
                       filename = TwoDaySuite_path*"strong_wind_no_rotation/instantaneous_statistics.jld2",
                       stressed = true,
                       rotating = false,
                          first = 13,
                           last = nothing),

                    "2d_strong_wind_weak_cooling" => (
                       filename = TwoDaySuite_path*"strong_wind_weak_cooling/instantaneous_statistics.jld2",
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),

                    "2d_weak_wind_strong_cooling" => (
                       filename = TwoDaySuite_path*"weak_wind_strong_cooling/instantaneous_statistics.jld2",
                       stressed = true,
                       rotating = true,
                          first = 13,
                           last = nothing),
                )

# https://engaging-web.mit.edu/~alir/lesbrary/6DaySuite/
SixDaySuite_path = "/Users/adelinehillier/.julia/dev/6DaySuite/"
SixDaySuite = OrderedDict(
                 "6d_free_convection" => (
                     filename = SixDaySuite_path*"free_convection/instantaneous_statistics.jld2",
                     stressed = false,
                     rotating = true,
                        first = 13, # cut out the first 2 hours
                         last = nothing),

                 "6d_strong_wind" => (
                     filename = SixDaySuite_path*"strong_wind/instantaneous_statistics.jld2",
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),

                 "6d_strong_wind_no_rotation" => (
                     filename = SixDaySuite_path*"strong_wind_no_rotation/instantaneous_statistics.jld2",
                     stressed = true,
                     rotating = false,
                        first = 13,
                         last = nothing),

                  "6d_strong_wind_weak_cooling" => (
                     filename = SixDaySuite_path*"strong_wind_weak_cooling/instantaneous_statistics.jld2",
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),

                  "6d_weak_wind_strong_cooling" => (
                     filename = SixDaySuite_path*"weak_wind_strong_cooling/instantaneous_statistics.jld2",
                     stressed = true,
                     rotating = true,
                        first = 13,
                         last = nothing),
              )

GeneralStrat_path = "/Users/adelinehillier/.julia/dev/8DayLinearStrat/"
GeneralStrat = OrderedDict(
                   "general_strat_4" => (
                       filename = GeneralStrat_path*"general_strat_4/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 37, # cut out the first 6 hours
                           last = 288), # 2 days -- mixed layer depth reaches about 75 meters

                   "general_strat_8" => (
                       filename = GeneralStrat_path*"general_strat_8/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = 648), # 4 days -- mixed layer depth reaches about 75 meters

                   "general_strat_16" => (
                       filename = GeneralStrat_path*"general_strat_16/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = nothing),

                    "general_strat_32" => (
                       filename = GeneralStrat_path*"general_strat_32/instantaneous_statistics.jld2",
                       stressed = false,
                       rotating = true,
                          first = 13,
                           last = nothing),
                )
