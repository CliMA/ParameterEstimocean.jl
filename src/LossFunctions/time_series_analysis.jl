#
# Time analysis
#

struct TimeSeriesAnalysis{T, D, A}
        time :: T
        data :: D
    analysis :: A
end

TimeSeriesAnalysis(time, analysis) = TimeSeriesAnalysis(time, zeros(length(time)), analysis)

struct TimeAverage end

# Use trapz integral to compute time average of data in case times are not evenly spaced
@inline (::TimeAverage)(data, time) = trapz(data, time) / (time[end] - time[1])

#
# Ensemble time analysis
#

struct EnsembleTimeSeriesAnalysis{T, D, A}
        time :: T
        data :: D
    analysis :: A
end

# Ensemble approach assumes evenly spaced time steps
EnsembleTimeSeriesAnalysis(time, ensemble_size) = EnsembleTimeSeriesAnalysis(time, zeros(ensemble_size, length(time)), data -> mean(data, dims=2))