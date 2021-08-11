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
# Profile analysis
#

"""
    struct ValueProfileAnalysis{D, A}

A type for doing analyses on a discrepancy profile located
at cell centers. Defaults to taking the mean square difference between
the model and data coarse-grained to the model grid.
"""
struct ValueProfileAnalysis{D, A}
    discrepancy :: D
       analysis :: A
end

ValueProfileAnalysis(grid; analysis=mean) = ValueProfileAnalysis(CenterField(grid), analysis)
ValueProfileAnalysis(; analysis=mean) = ValueProfileAnalysis(nothing, analysis)
on_grid(profile::ValueProfileAnalysis, grid) = ValueProfileAnalysis(grid; analysis=profile.analysis)

"""
    struct GradientProfileAnalysis{D, A}

A type for combining discreprancy between the fields and field gradients.
Defaults to taking the mean square difference between
the model and data coarse-grained to the model grid.
"""
mutable struct GradientProfileAnalysis{D, G, F, W, A}
     ϵ :: D
    ∇ϵ :: G
    ∇ϕ :: F
    gradient_weight :: W
    value_weight :: W
    analysis :: A
end

GradientProfileAnalysis(grid; analysis=mean, gradient_weight=1.0, value_weight=1.0) =
    GradientProfileAnalysis(CenterField(grid), FaceField(grid), FaceField(grid),
                            gradient_weight, value_weight, analysis)

GradientProfileAnalysis(; analysis=mean, gradient_weight=1.0, value_weight=1.0) =
    GradientProfileAnalysis(nothing, nothing, nothing, gradient_weight, value_weight, analysis)

function on_grid(profile::GradientProfileAnalysis, grid)
    return GradientProfileAnalysis(grid;
                                          analysis = profile.analysis,
                                   gradient_weight = profile.gradient_weight,
                                      value_weight = profile.value_weight)
end
