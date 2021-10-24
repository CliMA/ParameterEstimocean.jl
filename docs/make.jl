pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add OceanTurbulenceParameterEstimation to environment stack

using
  Documenter,
  Literate,
  Plots,  # so that Literate.jl does not capture precompilation output
  OceanTurbulenceParameterEstimation
  
# Gotta set this environment variable when using the GR run-time on CI machines.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

examples = [
  # "convective_adjustment_perfect_model_calibration.jl",
  # "convective_adjustment_perfect_model_calibration_uki.jl",
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
      canonical = "https://adelinehillier.github.io/OceanTurbulenceParameterEstimation/dev/",
)

pages = [
    "Home" => "index.md",
    "Installation Instructions" => "installation_instructions.md",
    #=
    "Examples" => [ 
        "literated/convective_adjustment_perfect_model_calibration.md",
        "literated/convective_adjustment_perfect_model_calibration_uki.md",
        ],
    =#
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
     # strict = true,
      clean = true,
  checkdocs = :exports
)

deploydocs(        repo = "github.com/adelinehillier/OceanTurbulenceParameterEstimation.git",
                versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
            push_preview = true
)
