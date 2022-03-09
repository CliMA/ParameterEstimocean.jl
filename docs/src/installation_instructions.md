# Installation instructions

You can install the latest version of OceanLearning.jl via the built-in
package manager (by pressing `]` in the Julia REPL command prompt) to add the package and also to 
instantiate/build all the required dependencies

To install, use Julia's  built-in package manager (accessed by pressing `]` in the Julia REPL
command prompt) to add the package and also to instantiate/build all the required dependencies.
The package is not yet included in Julia's official registry so you need to install via Github.

To install a tagged version of the package, e.g., v0.6.0, use

```julia
julia>]
(v1.6) pkg> add https://github.com/CliMA/OceanLearning.jl#v0.6.0
(v1.6) pkg> instantiate
```

Alternatively, if you'd like to be in the bleeding edge of the package's latest developments you may
install the version on the `#main` branch (or any other branch or commit), e.g.

```julia
julia>]
(v1.6) pkg> add https://github.com/CliMA/OceanLearning.jl#main
(v1.6) pkg> instantiate
```
