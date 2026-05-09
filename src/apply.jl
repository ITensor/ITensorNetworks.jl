using .BaseExtensions: maybe_real
using Graphs: has_edge, ne
using ITensors.NDTensors: scalartype
using ITensors: ITensors, ITensor, Index, Ops, apply, commonind, commoninds, contract, dag,
    denseblocks, factorize, factorize_svd, hascommoninds, hasqns, isdiag, noprime, prime,
    replaceind, replaceinds, tags, unioninds, uniqueinds
using LinearAlgebra: eigen, norm, qr, svd
using NamedGraphs: NamedEdge

# Reduced simple-update on a 2-site gate `o`. Assumes `envs` is a product
# environment: each `env` shares indices with exactly one of `ψ[v⃗[1]]` or
# `ψ[v⃗[2]]` and is a 2-leg matrix between an index of that tensor and its
# prime. Builds sqrt-envs and inverse-sqrt-envs via `map_eigvals`, QR-reduces
# each endpoint into a small bond tensor, applies the gate, factor-SVDs back,
# then unwinds the inverse sqrt envs.
function simple_update_bp(
        o::Union{NamedEdge, ITensor}, ψ, v⃗; envs, callback = Returns(nothing), apply_kwargs...
    )
    cutoff = 10 * eps(real(scalartype(ψ)))
    envs_v1 = filter(env -> hascommoninds(env, ψ[v⃗[1]]), envs)
    envs_v2 = filter(env -> hascommoninds(env, ψ[v⃗[2]]), envs)
    @assert all(ndims(env) == 2 for env in vcat(envs_v1, envs_v2))
    sqrt_envs_v1 = [
        ITensorsExtensions.map_eigvals(
                sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v1
    ]
    sqrt_envs_v2 = [
        ITensorsExtensions.map_eigvals(
                sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v2
    ]
    inv_sqrt_envs_v1 = [
        ITensorsExtensions.map_eigvals(
                inv ∘ sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v1
    ]
    inv_sqrt_envs_v2 = [
        ITensorsExtensions.map_eigvals(
                inv ∘ sqrt, env, inds(env)[1], inds(env)[2]; cutoff, ishermitian = true
            ) for env in envs_v2
    ]
    ψᵥ₁ = contract([ψ[v⃗[1]]; sqrt_envs_v1])
    ψᵥ₂ = contract([ψ[v⃗[2]]; sqrt_envs_v2])
    sᵥ₁ = siteinds(ψ, v⃗[1])
    sᵥ₂ = siteinds(ψ, v⃗[2])
    Qᵥ₁, Rᵥ₁ = qr(ψᵥ₁, uniqueinds(uniqueinds(ψᵥ₁, ψᵥ₂), sᵥ₁))
    Qᵥ₂, Rᵥ₂ = qr(ψᵥ₂, uniqueinds(uniqueinds(ψᵥ₂, ψᵥ₁), sᵥ₂))
    rᵥ₁ = commoninds(Qᵥ₁, Rᵥ₁)
    rᵥ₂ = commoninds(Qᵥ₂, Rᵥ₂)
    oR = apply(o, Rᵥ₁ * Rᵥ₂)
    e = v⃗[1] => v⃗[2]
    singular_values! = Ref(ITensor())
    Rᵥ₁, Rᵥ₂, spec = factorize_svd(
        oR,
        unioninds(rᵥ₁, sᵥ₁);
        ortho = "none",
        tags = edge_tag(e),
        singular_values!,
        apply_kwargs...
    )
    callback(; singular_values = singular_values![], truncation_error = spec.truncerr)
    Qᵥ₁ = contract([Qᵥ₁; dag.(inv_sqrt_envs_v1)])
    Qᵥ₂ = contract([Qᵥ₂; dag.(inv_sqrt_envs_v2)])
    ψᵥ₁ = Qᵥ₁ * Rᵥ₁
    ψᵥ₂ = Qᵥ₂ * Rᵥ₂
    return ψᵥ₁, ψᵥ₂
end

function ITensors.apply(
        o::Union{NamedEdge, ITensor},
        ψ::AbstractITensorNetwork;
        envs = ITensor[],
        normalize = false,
        ortho = false,
        callback = Returns(nothing),
        apply_kwargs...
    )
    ψ = copy(ψ)
    v⃗ = neighbor_vertices(ψ, o)
    if length(v⃗) == 1
        if ortho
            ψ = tree_orthogonalize(ψ, v⃗[1])
        end
        oψᵥ = apply(o, ψ[v⃗[1]])
        if normalize
            oψᵥ ./= norm(oψᵥ)
        end
        setindex_preserve_graph!(ψ, oψᵥ, v⃗[1])
    elseif length(v⃗) == 2
        envs = Vector{ITensor}(envs)
        if !iszero(ne(ITensorNetwork(envs)))
            error(
                "`apply` requires a product environment (`envs` with no shared edges); " *
                    "got `envs` with $(ne(ITensorNetwork(envs))) internal edges. " *
                    "Contract `envs` to product form before calling."
            )
        end
        e = v⃗[1] => v⃗[2]
        if !has_edge(ψ, e)
            error("Vertices where the gates are being applied must be neighbors for now.")
        end
        if ortho
            ψ = tree_orthogonalize(ψ, v⃗[1])
        end
        ψᵥ₁, ψᵥ₂ = simple_update_bp(o, ψ, v⃗; envs, callback, apply_kwargs...)
        if normalize
            ψᵥ₁ ./= norm(ψᵥ₁)
            ψᵥ₂ ./= norm(ψᵥ₂)
        end
        setindex_preserve_graph!(ψ, ψᵥ₁, v⃗[1])
        setindex_preserve_graph!(ψ, ψᵥ₂, v⃗[2])
    elseif length(v⃗) < 1
        error("Gate being applied does not share indices with tensor network.")
    elseif length(v⃗) > 2
        error("Gates with more than 2 sites is not supported yet.")
    end
    return ψ
end

function ITensors.apply(
        o⃗::Union{Vector{NamedEdge}, Vector{ITensor}},
        ψ::AbstractITensorNetwork;
        normalize = false,
        ortho = false,
        apply_kwargs...
    )
    o⃗ψ = ψ
    for oᵢ in o⃗
        o⃗ψ = apply(oᵢ, o⃗ψ; normalize, ortho, apply_kwargs...)
    end
    return o⃗ψ
end

function ITensors.apply(
        o⃗::Scaled,
        ψ::AbstractITensorNetwork;
        cutoff = nothing,
        normalize = false,
        ortho = false,
        apply_kwargs...
    )
    return maybe_real(Ops.coefficient(o⃗)) *
        apply(Ops.argument(o⃗), ψ; cutoff, maxdim, normalize, ortho, apply_kwargs...)
end

function ITensors.apply(
        o⃗::Prod, ψ::AbstractITensorNetwork; normalize = false, ortho = false, apply_kwargs...
    )
    o⃗ψ = ψ
    for oᵢ in o⃗
        o⃗ψ = apply(oᵢ, o⃗ψ; normalize, ortho, apply_kwargs...)
    end
    return o⃗ψ
end

function ITensors.apply(
        o::Op, ψ::AbstractITensorNetwork; normalize = false, ortho = false, apply_kwargs...
    )
    return apply(ITensor(o, siteinds(ψ)), ψ; normalize, ortho, apply_kwargs...)
end
