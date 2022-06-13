pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add ParameterEstimocean to environment stack

using
  Documenter,
  Literate,
  CairoMakie,  # so that Literate.jl does not capture precompilation output or warnings
  Distributions,
  ParameterEstimocean

# Gotta set this environment variable when using the GR run-time on CI machines.
# This happens when examples, e.g., use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

to_be_literated = [
  "intro_to_observations.jl",
  "intro_to_inverse_problems.jl",
  "exploring_priors.jl",
  "perfect_convective_adjustment_calibration.jl",
  "single_case_lesbrary_ri_based_calibration.jl",
  "multi_case_lesbrary_ri_based_calibration.jl",
  # "perfect_baroclinic_adjustment_calibration.jl"
]

for file in to_be_literated
    filepath = joinpath(EXAMPLES_DIR, file)
    Literate.markdown(filepath, OUTPUT_DIR; flavor = Literate.DocumenterFlavor())
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
      canonical = "https://clima.github.io/ParameterEstimocean/dev/",
)

pages = [
    "Home" => "index.md",
    "Installation Instructions" => "installation_instructions.md",

    "Intro to observations" => "literated/intro_to_observations.md",
    "Intro to inverse problems" => "literated/intro_to_inverse_problems.md",
    "Exploring Prior distributions" => "literated/exploring_priors.md",

    "Examples" => [ 
        "literated/perfect_convective_adjustment_calibration.md",
        "literated/single_case_lesbrary_ri_based_calibration.md",
        "literated/multi_case_lesbrary_ri_based_calibration.md",
        # "literated/perfect_baroclinic_adjustment_calibration.md"
        ],
    
    "Library" => [ 
        "Contents"       => "library/outline.md",
        "Public"         => "library/public.md",
        "Private"        => "library/internals.md",
        "Function index" => "library/function_index.md",
        ],
]

makedocs(
   sitename = "ParameterEstimocean.jl",
    modules = [ParameterEstimocean],
     format = format,
      pages = pages,
    doctest = true,
     strict = true,
      clean = true,
  checkdocs = :exports
)

withenv("GITHUB_REPOSITORY" => "CliMA/ParameterEstimoceanDocumentation") do
    deploydocs(        repo = "github.com/CliMA/ParameterEstimoceanDocumentation.git",
                   versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
                  forcepush = true,
                  devbranch = "main",
               push_preview = true
    )
end
