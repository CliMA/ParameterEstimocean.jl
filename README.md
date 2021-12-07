# OceanTurbulenceParameterEstimation.jl

A Julia package designed to leverage [Oceananigans.jl](http://github.com/CliMA/Oceananigans.jl/) and [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl) to allow for calibration of ocean turbulence parametrizations.

Continuous Integration: [![Build Status](https://github.com/CliMA/OceanTurbulenceParameterEstimation.jl/workflows/CI/badge.svg)](https://github.com/CliMA/OceanTurbulenceParameterEstimation.jl/actions?query=workflow%3ACI+branch%3Amaster)

Code Coverage: [![codecov](https://codecov.io/gh/CliMA/OceanTurbulenceParameterEstimation.jl/branch/main/graph/badge.svg?token=cPeTALmiPU)](https://codecov.io/gh/CliMA/OceanTurbulenceParameterEstimation.jl)

Stable Release Documentation: [![Build Status](https://img.shields.io/badge/documentation-stable%20release-blue)](https://clima.github.io/OceanTurbulenceParameterEstimation.jl/stable)

Latest Documentation: [![Build Status](https://img.shields.io/badge/documentation-in%20development-orange)](https://clima.github.io/OceanTurbulenceParameterEstimation.jl/dev)

Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5762810.svg)](https://doi.org/10.5281/zenodo.5762810)


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


## Citing

The code is citable via [zenodo](https://zenodo.org). Please cite as:

> Adeline Hillier, Gregory L. Wagner, and Navid C. Constantinou. (2021). CliMA/OceanTurbulenceParameterEstimation.jl: OceanTurbulenceParameterEstimation.jl v0.1.0 (Version v0.1.0). Zenodo. [http://doi.org/10.5281/zenodo.5762810](http://doi.org/10.5281/zenodo.5762810)

