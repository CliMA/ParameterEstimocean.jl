using Test
using ParameterEstimocean
using ParameterEstimocean.Utils: prettyvector

@test prettyvector([0, 1, 2]) == "[0, 1, 2]"
@test prettyvector(collect(0:20)) == "[0, 1, 2 … 20, 19, 18] (21 elements)"
