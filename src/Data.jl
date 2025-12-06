module DataType

using Random, LinearAlgebra, Statistics

"""
Generate a non-negative matrix X ∈ R^{m×n} by sampling non-negative factors W (m×r)
and H (r×n) and returning (X, W, H)

Optionally add Gaussian noise (clipped at 0 to keep non-negativity).
"""
function generate_synthetic_data(m::Int, n::Int, rank::Int=10, 
    noise_level::Float64=0.0, seed=nothing)
    
    # Only call if seed is given
    if seed !== nothing
        Random.seed!(seed)
    end

    W = rand(m, rank)
    H = rand(rank, n)
    X = W * H

    if noise_level > 0
        noise = similar(X)          # Create matrix with uninitialized values
        randn!(noise)               # Fill with Gaussian noise N(0,1)
        X .+= noise_level .* noise  # Add scaled noise
        @. X = max(X, 0.0)          # Clip to keep non-negative
    end

    return X, W, H
end
end

