# ParameterEstimocean.jl

A Julia package that leverages [Oceananigans.jl](http://github.com/CliMA/Oceananigans.jl/) and [EnsembleKalmanProcesses.jl](https://github.com/CliMA/EnsembleKalmanProcesses.jl) to calibrate ocean turbulence parametrizations.

ParameterEstimocean.jl is developed by the [Climate Modeling Alliance](https://clima.caltech.edu) and heroic external collaborators.

Continuous Integration: [![Build Status](https://github.com/CliMA/ParameterEstimocean.jl/workflows/CI/badge.svg)](https://github.com/CliMA/ParameterEstimocean.jl/actions?query=workflow%3ACI+branch%3Amaster)

Code Coverage: [![codecov](https://codecov.io/gh/CliMA/ParameterEstimocean.jl/branch/main/graph/badge.svg?token=cPeTALmiPU)](https://codecov.io/gh/CliMA/ParameterEstimocean.jl)

Stable Release Documentation: [![Build Status](https://img.shields.io/badge/documentation-stable%20release-blue)](https://clima.github.io/ParameterEstimoceanDocumentation/stable)

Latest Documentation: [![Build Status](https://img.shields.io/badge/documentation-in%20development-orange)](https://clima.github.io/ParameterEstimoceanDocumentation/dev)

Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5762810.svg)](https://doi.org/10.5281/zenodo.5762810)


## Installation

To install, use Julia's  built-in package manager (accessed by pressing `]` in the Julia REPL command prompt) to add the package and also to instantiate/build all the required dependencies. To install the latest tagged version of the package, use

```julia
julia> ]
(v1.6) pkg> add ParameterEstimocean
(v1.6) pkg> instantiate
```

Alternatively, if you'd like to be in the cutting-edge of the package's latest developments you may
install the version on the `#main` branch (or any other branch or commit), e.g.

```julia
julia> ]
(v1.6) pkg> add ParameterEstimocean#main
(v1.6) pkg> instantiate
```


## Citing

The code is citable via [zenodo](https://zenodo.org). Please cite as:

> Adeline Hillier, Gregory L. Wagner, and Navid C. Constantinou. (2022). CliMA/ParameterEstimocean.jl: ParameterEstimocean.jl v0.14.1 (Version v0.14.1). Zenodo. [http://doi.org/10.5281/zenodo.5762810](http://doi.org/10.5281/zenodo.5762810)
