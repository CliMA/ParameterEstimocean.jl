# for LEScase in values(training_simulations)
#     fields = !(LEScase.stressed) ? (:T, :e) :
#              !(LEScase.rotating) ? (:T, :U, :e) :
#                                    (:T, :U, :V, :e)
#
#     data = TruthData(LEScase.filename)
#
#     targets = (LEScase.first, LEScase.last)
#     μ = profile_mean(data, field_name, targets)
#     σ = sqrt(mean_variance(data, field_name, targets))
#     normalize(Φ) = (Φ .- μ) ./ σ
#     return normalize
# end

# normalize_function = Dict()
# for LEScase in values(training_simulations)
#     fields = !(LEScase.stressed) ? (:T, :e) :
#              !(LEScase.rotating) ? (:T, :U, :e) :
#                                    (:T, :U, :V, :e)
#
#     data = TruthData(LEScase.filename)
#
#     targets = (LEScase.first, LEScase.last)
#     μ = profile_mean(data, field_name, targets)
#     σ = sqrt(mean_variance(data, field_name, targets))
#     normalize(Φ) = (Φ .- μ) ./ σ
#     normalize_function[data.name] = normalize
# end

# normalize_function = get_normalization_functions(FourDaySuite)
# normalize_function[:U](parent(ce.calibration.loss.batch[1].data.U[end].data))

function get_normalization_functions_each_simulation(LESdata)
    normalize_function = Dict()
    for LEScase in values(LESdata)
        fields = !(LEScase.stressed) ? (:T, :e) :
                 !(LEScase.rotating) ? (:T, :U, :e) :
                                       (:T, :U, :V, :e)

        data = TruthData(LEScase.filename)
        first = LEScase.first
        last = LEScase.last == nothing ? length(data) : LEScase.last
        targets = (first, last)

        normalize_function[data.name] = Dict()
        for field_name in fields
            μ = profile_mean(data, field_name; targets=targets)
            σ = sqrt(mean_variance(data, field_name, targets))
            normalize(Φ) = (Φ .- μ) ./ σ
            normalize_function[data.name][field_name] = normalize
        end
    end
    return normalize_function
end

function get_normalization_functions(loss::BatchedLossContainer; data_indices = 28:126)
    normalize_function = Dict()
    for simulation in ce.calibration.loss.batch
        data = simulation.data
        case = data.name
        normalize_function[case] = Dict()
        loss = simulation.loss
        fields = loss.fields
        targets = loss.targets

        for field in fields
            μ = OceanTurbulenceParameterEstimation.profile_mean(data, field; targets=targets, indices=data_indices)
            # σ = sqrt(mean_variance(data, field, targets, indices=data_indices))
            σ = mean_std(data, field; targets=targets, indices=data_indices)

            # if field != :T; μ=0; end
            normalize(Φ) = (Φ .- μ) ./ σ
            normalize_function[case][field] = normalize
        end
    end
    return normalize_function
end

# function get_normalization_functions(loss::BatchedLossFunction; data_indices = 40:127)
#     normalize_function = Dict()
#
#     for field in (:T, :U, :V, :e)
#         normalize_function[field] = Dict()
#         μs = []
#         σs = []
#         for simulation in ce.calibration.loss.batch
#             data = simulation.data
#             case = data.name
#             loss = simulation.loss
#             targets = loss.targets
#             fields = loss.fields
#             if field in fields
#                 push!(μs, OceanTurbulenceParameterEstimation.profile_mean(data, field; targets=targets, indices=data_indices))
#                 # σ = sqrt(mean_variance(data, field, targets))
#                 push!(σs, mean_std(data, field; targets=targets, indices=data_indices))
#             end
#         end
#         normalize(Φ) = (Φ .- mean(μs)) ./ mean(σs)
#         normalize_function[field] = normalize
#     end
#     return normalize_function
# end


# layout = @layout [c; d]
# Plots.plot(a,b,size=(1000,500), margins=5*Plots.mm, layout = layout, xlabel="")
# Plots.savefig("EKI/forward_map_output")



# normalize_function = Dict()
# for simulation in ce.calibration.loss.batch
#     data = simulation.data
#     case = data.name
#     normalize_function[case] = Dict()
#     loss = simulation.loss
#     fields = loss.fields
#     targets = loss.targets
#
#     println(case, fields)
#
#     for field in fields
#         μ = OceanTurbulenceParameterEstimation.profile_mean(data, field; targets=targets)
#         # σ = sqrt(mean_variance(data, field, targets))
#         σ = sqrt(max_variance(data, field; targets=targets))
#         normalize(Φ) = (Φ .- μ) ./ σ
#         normalize_function[case][field] = normalize
#     end
# end
# return normalize_function

a = [-4.942294253851287e-5, -9.710021913633682e-5, -0.00016448897804366425, -0.0003282837278675288, -0.0005497726378962398, -0.0007353588589467108, -0.000766437326092273, -0.0006677638739347458, -0.000670047098537907, -0.0005327400285750628, 0.0024554537376388907, 0.014912797138094902, 0.03547245543450117, 0.055420275777578354, 0.07403376698493958, 0.09250877425074577, 0.11088467389345169, 0.12782885879278183, 0.142448328435421, 0.15616662800312042, 0.16957376897335052, 0.1823633313179016, 0.1935504898428917, 0.20413243770599365, 0.21547429263591766, 0.2270088642835617, 0.23757176101207733, 0.24661502987146378, 0.2552718371152878, 0.26404528319835663, 0.2719387710094452, 0.2793842554092407, 0.2868467718362808, 0.29382631182670593, 0.30060000717639923, 0.3070996254682541, 0.3133727163076401, 0.3198990225791931, 0.3263692110776901, 0.33267226815223694, 0.3386473059654236, 0.3454121798276901, 0.35282863676548004, 0.36104652285575867]

b = [0.165313277970034, 0.09611743204808944, -0.001686438285295429, -0.23940799204016913, -0.5608632760960883, -0.8302116213333761, -0.8753169834167609, -0.7321084472062372, -0.7354221779626778, -0.5361432019991834, 3.8007361289249584, 21.880552679815153, 51.71956717903865, 80.67059797263056, 107.68506611263379, 134.49854723814505, 161.16818969863385, 185.7599300670529, 206.97772271460474, 226.88761293767453, 246.34590684685674, 264.9078853941602, 281.14423429742186, 296.5022179353269, 312.96308361816017, 329.70364640740377, 345.0339804086781, 358.1588206528749, 370.7227744473112, 383.4560106279245, 394.9121298809223, 405.71804473399527, 416.5486788163752, 426.6783508162551, 436.5092726554644, 445.942416030419, 455.04679149292866, 464.5186681484509, 473.9090989443318, 483.05696554431853, 491.72876530726734, 501.5468841860657, 512.3106703788675, 524.2376011996804]
#
# plot(a)
#
# plot(b)
