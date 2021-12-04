# OceanTurbulenceParameterEstimation.jl

A Julia package designed to leverage [Oceananigans.jl](http://github.com/CliMA/Oceananigans.jl/) and [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl) to allow for calibration of ocean turbulence parametrizations.

Continuous Integration: [![Build Status](https://github.com/CliMA/OceanTurbulenceParameterEstimation.jl/workflows/CI/badge.svg)](https://github.com/CliMA/OceanTurbulenceParameterEstimation.jl/actions?query=workflow%3ACI+branch%3Amaster)

Code Coverage: [![codecov](https://codecov.io/gh/CliMA/OceanTurbulenceParameterEstimation.jl/branch/main/graph/badge.svg?token=cPeTALmiPU)](https://codecov.io/gh/CliMA/OceanTurbulenceParameterEstimation.jl)

Latest Documentation: [![Build Status](https://img.shields.io/badge/documentation-in%20development-orange)](https://clima.github.io/OceanTurbulenceParameterEstimation.jl/dev)


## Installation

To install, use Julia's  built-in package manager (accessed by pressing `]` in the Julia REPL command prompt) to add the package and also to instantiate/build all the required dependencies

```julia
julia>]
(v1.6) pkg> add OceanTurbulenceParameterEstimation
(v1.6) pkg> instantiate
```

Alternatively, if you'd like to be in the bleeding edge of the package's latest developments you may
install the version on the `#main` branch (or any other branch or commit), e.g.

```julia
julia>]
(v1.6) pkg> add OceanTurbulenceParameterEstimation#main
(v1.6) pkg> instantiate
```
