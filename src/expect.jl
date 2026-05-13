using ITensors: Op, contract, op, which_op

default_expect_alg() = "bp"

function expect(ψIψ::AbstractFormNetwork, op::Op; kwargs...)
    v = only(op.sites)
    ψIψ_v = ψIψ[operator_vertex(ψIψ, v)]
    s = commonind(ψIψ[ket_vertex(ψIψ, v)], ψIψ_v)
    operator = ITensors.op(op.which_op, s)
    ∂ψIψ_∂v = environment(ψIψ, operator_vertices(ψIψ, [v]); kwargs...)
    numerator_ts = vcat(∂ψIψ_∂v, operator)
    denominator_ts = vcat(∂ψIψ_∂v, ψIψ_v)
    numerator_seq = contraction_sequence(numerator_ts; alg = "optimal")
    denominator_seq = contraction_sequence(denominator_ts; alg = "optimal")
    numerator = contract(numerator_ts; sequence = numerator_seq)[]
    denominator = contract(denominator_ts; sequence = denominator_seq)[]

    return numerator / denominator
end

function expect(
        alg::Algorithm,
        ψ::AbstractITensorNetwork,
        ops;
        (cache!) = nothing,
        update_cache = isnothing(cache!),
        cache_update_kwargs = (;),
        cache_construction_kwargs = (;),
        kwargs...
    )
    ψIψ = QuadraticFormNetwork(ψ)
    if isnothing(cache!)
        cache! = Ref(initialize_cache(alg, ψIψ; cache_construction_kwargs...))
    end

    if update_cache
        cache![] = update(cache![]; cache_update_kwargs...)
    end

    return map(op -> expect(ψIψ, op; alg, cache!, update_cache = false, kwargs...), ops)
end

function expect(alg::Algorithm"exact", ψ::AbstractITensorNetwork, ops; kwargs...)
    ψIψ = QuadraticFormNetwork(ψ)
    return map(op -> expect(ψIψ, op; alg, kwargs...), ops)
end

"""
    expect(ψ::AbstractITensorNetwork, op::Op; alg="bp", kwargs...) -> Number

Compute the expectation value ⟨ψ|op|ψ⟩ / ⟨ψ|ψ⟩ for a single `ITensors.Op` object.

The default algorithm is belief propagation (`"bp"`); use `alg="exact"` for exact
contraction.

See also: [`expect(ψ, op::String)`](@ref).
"""
function expect(ψ::AbstractITensorNetwork, op::Op; alg = default_expect_alg(), kwargs...)
    return expect(Algorithm(alg), ψ, [op]; kwargs...)
end

"""
    expect(ψ::AbstractITensorNetwork, op::String, vertices; alg="bp", kwargs...) -> Dictionary

Compute local expectation values ⟨ψ|op_v|ψ⟩ / ⟨ψ|ψ⟩ for the operator named `op` at each
vertex in `vertices`.

See [`expect(ψ, op::String)`](@ref) for full documentation.
"""
function expect(
        ψ::AbstractITensorNetwork, op::String, vertices; alg = default_expect_alg(), kwargs...
    )
    return expect(Algorithm(alg), ψ, [Op(op, vertex) for vertex in vertices]; kwargs...)
end

"""
    expect(ψ::AbstractITensorNetwork, op::String; alg="bp", kwargs...) -> Dictionary

Compute local expectation values ⟨ψ|op_v|ψ⟩ / ⟨ψ|ψ⟩ for the operator named `op` at every
vertex of `ψ`.

# Arguments

  - `ψ`: The tensor network state.
  - `op`: Name of the local operator (e.g. `"Sz"`, `"N"`, `"Sx"`), passed to `ITensors.op`.
  - `alg="bp"`: Contraction algorithm. `"bp"` uses belief propagation (efficient for
    loopy or large networks); `"exact"` performs full contraction.

# Keyword Arguments (alg="bp" only)

  - `cache!`: Optional `Ref` to a pre-built belief propagation cache. If provided,
    the cache is reused across multiple `expect` calls for efficiency.
  - `update_cache=true`: Whether to update the cache before computing expectation values.

# Returns

A `Dictionary` mapping each vertex of `ψ` to its expectation value.

See also: [`expect(ψ, op::String, vertices)`](@ref),
[`expect(operator, state::AbstractTreeTensorNetwork)`](@ref).
"""
function expect(
        ψ::AbstractITensorNetwork,
        op::String;
        alg = default_expect_alg(),
        kwargs...
    )
    return expect(ψ, op, vertices(ψ); alg, kwargs...)
end
