using Random, LinearAlgebra

# numerical gradient
function gradient(L, params; δ = 1e-4 .* ones(length(params)))
    ∇L = zeros(length(params))
    e = I + zeros(length(params), length(params))
    Lcurrent = L(params)
    for i in eachindex(params)
        up = L(params + δ[i] * e[i,:])
        ∇L[i] = (up - Lcurrent)/(δ[i])
    end
    return ∇L
end

# Define Method structs
struct RandomPlugin{S,T,V,U}
    priors::S
    fcalls::T
    seed::V
    progress::U
end

function RandomPlugin(priors, fcalls::Int; seed = 1234, progress = true)
    return RandomPlugin(priors, fcalls, 1234, true)
end

struct RandomLineSearch{I, T, B}
    linesearches::I
    linelength::I
    linebounds::T
    progress::B
    seed::I
end

function RandomLineSearch(linesearches, linelength)
    return RandomLineSearch(linesearches, linelength, (-0.1,1), true, 1234)
end
function RandomLineSearch(; linesearches = 10, linelength = 10, linebounds = (-0.1,1), progress = true, seed = 1234)
    return RandomLineSearch(linesearches, linelength, linebounds, progress, seed)
end

# Define Helper functions

# RandomPlugin
function priorloss(ℒ, fcalls, priors; seed = 1234, progress = true)
    Random.seed!(seed)
    losses = []
    vals = []
    for i in 1:fcalls
        guessparams = [rand.(priors)...]
        push!(vals, guessparams)
        currentloss = ℒ(guessparams)
        push!(losses, currentloss)
        # if progress
        #     println("iteration " * string(i))
        # end
    end
    return losses, vals
end

# LineSearch
function randomlinesearch(ℒ, ∇ℒ, bparams; linesearches = 10, linelength = 10, linebounds = (-0.1, 1.0), progress = true, seed = 1234)
    params = copy(bparams)
    Random.seed!(seed)
    for i in 1:linesearches
        αs = [rand(Uniform(linebounds...), linelength-1)..., 0.0]
        ∇L = ∇ℒ(params)
        LS = [ℒ(params - α * ∇L) for α in αs]
        b = argmin(LS)
        if progress
            println("best after line search ", i)
        end
        params .= params - αs[b] * ∇L
        if αs[b] == 0
            αs .= 0.5 .* αs
            if progress
                println("staying the same at ")
            end
        end
        if progress
            println((params, LS[b]))
        end
    end
    return params
end

# Define optimize functions
function optimize(ℒ, method::RandomPlugin; history = false, printresult = true)
    losses, vals = priorloss(ℒ, method.fcalls,
                       method.priors, seed = method.seed,
                       progress = method.progress)
    indmin = argmin(losses)
    if printresult
        println("The minimum loss is ", losses[indmin])
        println("The minimum argument is ", vals[indmin])
        println("This occured at iteration ", indmin)
    end
    if history == false
        return vals[indmin]
    else
        return vals, losses
    end
end
optimize(ℒ, initialguess, method::RandomPlugin; history = false, printresult = true) = optimize(ℒ, method::RandomPlugin; history = false, printresult = true)

function optimize(ℒ, ∇ℒ, params, method::RandomLineSearch; history = false, printresult = true)
    bestparams = randomlinesearch(ℒ, ∇ℒ, params; linesearches = method.linesearches,
                    linelength = method.linelength,
                    linebounds = method.linebounds,
                    progress = method.progress,
                    seed = method.seed)
    return bestparams
end
