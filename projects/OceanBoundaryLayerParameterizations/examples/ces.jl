using GaussianProcesses
using FileIO
using LinearAlgebra

# description = ""
# noise_cov_name = "noise_covariance_0001"
# file = "calibrate_convadj_to_lesbrary/loss_landscape_$(description).jld2"

# G = load(file)["G"]
# Φ1 = load(file)[noise_cov_name*"/Φ1"]
# Φ2 = load(file)[noise_cov_name*"/Φ2"]

# directory = "QuickCES/"
# isdir(directory) || mkdir(directory)

# pvalues = Dict(
#     :convective_κz => collect(0.075:0.025:1.025),
#     :background_κz => collect(0e-4:0.25e-4:10e-4),
# )

# ni = length(pvalues[:convective_κz])
# nj = length(pvalues[:background_κz])

Φ = Φ1 .+ Φ2

x = hcat([[pvalues[:convective_κz][i], pvalues[:background_κz][j]] for i = 1:ni, j = 1:nj]...)

not_nan_indices = findall(.!isnan.(Φ))
Φ = Φ[not_nan_indices]
x = x[:, not_nan_indices]

OceanLearning.Transformations.normalize!(Φ, ZScore(mean(Φ), var(Φ)))


using OceanLearning.Transformations: ZScore, normalize!
using Statistics


ces_directory = joinpath(directory, "QuickCES/")
isdir(ces_directory) || mkdir(ces_directory)


# MZero
#  * Candidate solution
#     Final objective value:     -7.398974e+03

mZero = MeanZero()
# kern = Matern(5 / 2, [0.0 for _ in 1:ni*nj], 0.0) + SE(0.0, 0.0)
kern = Matern(5 / 2, [0.0, 0.0], 0.0)
gp = GP(x, Φ, mZero, kern, -2.0)

optimize!(gp)

xs = x[1, :]
ys = x[2, :]
Φ_predicted = predict_f(gp, x)[1]

using Plots
p = Plots.plot(gp)
Plots.savefig(p, joinpath(directory, "hello.pdf"))

plot_contour(eki, xs, ys, Φ_predicted, "GP_emulated", ces_directory; zlabel = "Φ", plot_minimizer=true, plot_scatters=false, title="GP-Emulated EKI Objective, Φ")
plot_contour(eki, xs, ys, Φ, "Original", ces_directory; zlabel = "Φ", plot_minimizer=true, plot_scatters=false, title="EKI Objective, Φ")