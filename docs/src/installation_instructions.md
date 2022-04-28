# Installation instructions

To install, use Julia's  built-in package manager (accessed by pressing `]` in the Julia REPL command prompt) to add the package and also to instantiate/build all the required dependencies. To install the latest tagged version of the package, use

```julia
julia>]
(v1.6) pkg> add ParameterEstimocean
(v1.6) pkg> instantiate
```

Alternatively, if you'd like to be in the bleeding edge of the package's latest developments you may
install the version on the `#main` branch (or any other branch or commit), e.g.

```julia
julia>]
(v1.6) pkg> add ParameterEstimocean#main
(v1.6) pkg> instantiate
```
