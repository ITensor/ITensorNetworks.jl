using ITensors: inner, scalar
using LinearAlgebra: norm, norm_sqr

default_contract_alg(tns::Tuple) = "bp"

"""
    inner(ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; alg="bp", kwargs...) -> Number

Compute the inner product ⟨ϕ|ψ⟩ by contracting the combined bra-ket network.

# Keyword Arguments
- `alg="bp"`: Contraction algorithm. `"bp"` uses belief propagation (default, efficient
  for large or loopy networks); `"exact"` uses full contraction with an optimized sequence.

See also: [`loginner`](@ref ITensorNetworks.loginner), `norm`, [`inner(ϕ, A, ψ)`](@ref ITensorNetworks.inner).
"""
function ITensors.inner(
        ϕ::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        alg = default_contract_alg((ϕ, ψ)),
        kwargs...
    )
    return inner(Algorithm(alg), ϕ, ψ; kwargs...)
end

"""
    inner(ϕ::AbstractITensorNetwork, A::AbstractITensorNetwork, ψ::AbstractITensorNetwork; alg="bp", kwargs...) -> Number

Compute the matrix element ⟨ϕ|A|ψ⟩ where `A` is a tensor network operator.

# Keyword Arguments
- `alg="bp"`: Contraction algorithm. `"bp"` (default) or `"exact"`.

See also: [`inner(ϕ, ψ)`](@ref).
"""
function ITensors.inner(
        ϕ::AbstractITensorNetwork,
        A::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        alg = default_contract_alg((ϕ, A, ψ)),
        kwargs...
    )
    return inner(Algorithm(alg), ϕ, A, ψ; kwargs...)
end

function ITensors.inner(
        alg::Algorithm"exact",
        ϕ::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        sequence = nothing,
        contraction_sequence_kwargs = (;),
        kwargs...
    )
    tn = inner_network(ϕ, ψ; kwargs...)
    if isnothing(sequence)
        sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
    end
    return scalar(tn; sequence)
end

function ITensors.inner(
        alg::Algorithm"exact",
        ϕ::AbstractITensorNetwork,
        A::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        sequence = nothing,
        contraction_sequence_kwargs = (;),
        kwargs...
    )
    tn = inner_network(ϕ, A, ψ; kwargs...)
    if isnothing(sequence)
        sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
    end
    return scalar(tn; sequence)
end

"""
    loginner(ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; alg="bp", kwargs...) -> Number

Compute `log(⟨ϕ|ψ⟩)` in a numerically stable way by accumulating logarithms during
contraction rather than computing the inner product directly.

Useful when the inner product would overflow or underflow in floating-point arithmetic.

# Keyword Arguments
- `alg="bp"`: Contraction algorithm, `"bp"` (default) or `"exact"`.

See also: [`inner`](@ref ITensorNetworks.inner), `lognorm`.
"""
function loginner(
        ϕ::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        alg = default_contract_alg((ϕ, ψ)),
        kwargs...
    )
    return loginner(Algorithm(alg), ϕ, ψ; kwargs...)
end

function loginner(
        ϕ::AbstractITensorNetwork,
        A::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        alg = default_contract_alg((ϕ, A, ψ)),
        kwargs...
    )
    return loginner(Algorithm(alg), ϕ, A, ψ; kwargs...)
end

function loginner(
        alg::Algorithm"exact", ϕ::AbstractITensorNetwork, ψ::AbstractITensorNetwork; kwargs...
    )
    return log(inner(alg, ϕ, ψ); kwargs...)
end

function loginner(
        alg::Algorithm"exact",
        ϕ::AbstractITensorNetwork,
        A::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        kwargs...
    )
    return log(inner(alg, ϕ, A, ψ); kwargs...)
end

function loginner(
        alg::Algorithm,
        ϕ::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        dual_link_index_map = sim,
        kwargs...
    )
    tn = inner_network(ϕ, ψ; dual_link_index_map)
    return logscalar(alg, tn; kwargs...)
end

function loginner(
        alg::Algorithm,
        ϕ::AbstractITensorNetwork,
        A::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        dual_link_index_map = sim,
        kwargs...
    )
    tn = inner_network(ϕ, A, ψ; dual_link_index_map)
    return logscalar(alg, tn; kwargs...)
end

function ITensors.inner(
        alg::Algorithm,
        ϕ::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        dual_link_index_map = sim,
        kwargs...
    )
    tn = inner_network(ϕ, ψ; dual_link_index_map)
    return scalar(alg, tn; kwargs...)
end

function ITensors.inner(
        alg::Algorithm,
        ϕ::AbstractITensorNetwork,
        A::AbstractITensorNetwork,
        ψ::AbstractITensorNetwork;
        dual_link_index_map = sim,
        kwargs...
    )
    tn = inner_network(ϕ, A, ψ; dual_link_index_map)
    return scalar(alg, tn; kwargs...)
end

# TODO: rename `sqnorm` to match https://github.com/JuliaStats/Distances.jl,
# or `norm_sqr` to match `LinearAlgebra.norm_sqr`
LinearAlgebra.norm_sqr(ψ::AbstractITensorNetwork; kwargs...) = inner(ψ, ψ; kwargs...)

function LinearAlgebra.norm(ψ::AbstractITensorNetwork; kwargs...)
    return sqrt(abs(real(norm_sqr(ψ; kwargs...))))
end
