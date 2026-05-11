using ITensors: Trotter
function tebd(
        ℋ::Sum,
        ψ::AbstractITensorNetwork;
        β,
        Δβ,
        maxdim,
        cutoff,
        print_frequency = 10,
        ortho = false,
        kwargs...
    )
    𝒰 = exp(-Δβ * ℋ; alg = Trotter{2}())
    # Imaginary time evolution terms
    s = siteinds(ψ)
    u⃗ = Vector{ITensor}(𝒰, s)
    nsteps = Int(β ÷ Δβ)
    ψ = add_edges(ψ, edges(ψ))
    for step in 1:nsteps
        if step % print_frequency == 0
            @show step, (step - 1) * Δβ, β
        end
        ψ = apply(u⃗, ψ; cutoff, maxdim, normalize = true, ortho, kwargs...)
        if ortho
            for v in vertices(ψ)
                ψ = tree_orthogonalize(ψ, v)
            end
        end
    end
    return ψ
end
