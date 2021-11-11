pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add OceanTurbulenceParameterEstimation to environment stack

using
  Documenter,
  Literate,
  CairoMakie,  # so that Literate.jl does not capture precompilation output or warnings
  Distributions,
  OceanTurbulenceParameterEstimation
  
# Gotta set this environment variable when using the GR run-time on CI machines.
# This happens when examples, e.g., use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

examples = [
  "intro_to_observations.jl",
  "intro_to_inverse_problems.jl",
  "perfect_convective_adjustment_calibration.jl",
  "perfect_baroclinic_adjustment_calibration.jl"
]

for example in examples
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR; flavor = Literate.DocumenterFlavor())
end

#####
##### Build and deploy docs
#####

# Set up a timer to print a space ' ' every 240 seconds. This is to avoid CI
# timing out when building demanding Literate.jl examples.
Timer(t -> println(" "), 0, interval=240)

format = Documenter.HTML(
  collapselevel = 2,
     prettyurls = get(ENV, "CI", nothing) == "true",
      canonical = "https://clima.github.io/OceanTurbulenceParameterEstimation/dev/",
)

pages = [
    "Home" => "index.md",
    "Installation Instructions" => "installation_instructions.md",
    
    "Examples" => [ 
        "literated/intro_to_observations.md",
        "literated/intro_to_inverse_problems.md",
        "literated/perfect_convective_adjustment_calibration.md",
        "perfect_baroclinic_adjustment_calibration.md"
       ],
    
    "Library" => [ 
        "Contents"       => "library/outline.md",
        "Public"         => "library/public.md",
        "Private"        => "library/internals.md",
        "Function index" => "library/function_index.md",
        ],
]

makedocs(
   sitename = "OceanTurbulenceParameterEstimation.jl",
    modules = [OceanTurbulenceParameterEstimation],
     format = format,
      pages = pages,
    doctest = true,
     strict = true,
      clean = true,
  checkdocs = :exports
)

deploydocs(        repo = "github.com/CliMA/OceanTurbulenceParameterEstimation.jl",
               versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
              forcepush = true,
              devbranch = "main",
           push_preview = true
)
