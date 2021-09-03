## Optimizing TKE parameters
using TKECalibration2021
using Plots, PyPlot
using OceanTurbulenceParameterEstimation: visualize_predictions
using Dao
using Statistics
using CalibrateEmulateSample.ParameterDistributionStorage

LESdata = FourDaySuite # Calibration set
LESdata_validation = merge(TwoDaySuite, SixDaySuite) # Validation set
RelevantParameters = TKEParametersRiIndependent
ParametersToOptimize = TKEParametersRiIndependent

propertynames(initial_parameters)
initial_parameters
relative_weights_options = Dict(
                "all_T" => Dict(:T => 1.0, :U => 0.0, :V => 0.0, :e => 0.0),
                "mostly_T" => Dict(:T => 1.0, :U => 0.33, :V => 0.33, :e => 0.33),
                "uniform" => Dict(:T => 1.0, :U => 1.0, :V => 1.0, :e => 1.0)
)

# define closure here cause ParametersToOptimize has to be in the global scope
function loss_closure(loss)
        ℒ(parameters::ParametersToOptimize) = loss(parameters)
        ℒ(parameters::Vector) = loss(ParametersToOptimize([parameters...]))
        return ℒ
end

loss, initial_parameters = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize;
                                loss_closure = loss_closure,
                                relative_weights = relative_weights)

initial_parameters = ParametersToOptimize([0.1320799067908237, 0.21748565946199314, 0.051363488558909924, 0.5477193236638974, 0.8559038503413254, 3.681157252463703, 2.4855193201082426])
bounds, _ = get_bounds_and_variance(initial_parameters; stds_within_bounds = stds_within_bounds);
initial_parameters = set_prior_means_to_initial_parameters ? initial_parameters : ParametersToOptimize([mean.(bounds)...])

loss, initial_parameters = custom_tke_calibration(LESdata, RelevantParameters, ParametersToOptimize;
                                relative_weights = relative_weights)
strong_wind_weak_cooling_loss = loss.batch[2].loss

fields = strong_wind_weak_cooling_loss.fields
data = loss.batch[2].data
targets = strong_wind_weak_cooling_loss.targets

max_variances = [max_variance(data, field, targets) for field in fields]
mean_variances = [mean_variance(data, field, targets) for field in fields]


max_var_weights = [1/σ for σ in max_variances]
max_std_weights = sqrt.(max_var_weights)
mean_var_weights = [1/σ for σ in mean_variances]
mean_std_weights = sqrt.(mean_var_weights)

println(mean_variances)

strong_wind_weak_cooling_loss.weights
