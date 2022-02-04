module EnsembleKalmanUpdate

"""
    update_ensemble!(θ, y, G, Γη)

Update the parameter ensemble `θ`, given a vector of observations `y`,
forward map output `G`, and noise covariance `Γη`.


"""
function update_ensemble!(θ, y, G, Γη) 
    Nobs, Nens = size(G)

    ndims(y) == 1 && (y = reshape(y, length(y), 1))

    # Check dimensions in user-friendly manner
    size(y) == (Nobs, 1) || throw(ArgumentError("size(y) must be `size(G, 1), 1`."))
    size(θ, 2) == Nens || throw(ArgumentError("size(θ, 2) must be equal to size(G, 2)`."))

    # Compute noise first
    η = rand(MvNormal(Nobs, Γη), Nens)

    # Compute covariances
    Γᶿᴳ = cov(θ, G, dims=2, corrected=false)
    Γᴳᴳ = cov(G, G, dims=2, corrected=false)

    # Compute update
    Δθ = (Γᴳᴳ + η) \ (y + η - G)
    θ .+= Γᶿᴳ * Δθ

    return nothing
end

end # module

