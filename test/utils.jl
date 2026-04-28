# Test-only helpers extracted from `src/`. Lives at `test/utils.jl` so the
# `ITensorPkgSkeleton.runtests` runner doesn't auto-pick it up (it only globs
# `test_*.jl`). Each test file that needs these does `include("utils.jl")`
# inside its gensym module.

using DataGraphs: IsUnderlyingGraph, vertex_data
using Distributions: Distribution
using Graphs: edges, vertices
using ITensorNetworks: ITensorNetworks, ITensorNetwork, IndsNetwork
using ITensors.NDTensors: dim
using ITensors: ITensor, Index, itensor
using LinearAlgebra: normalize
using NamedGraphs.GraphsExtensions: incident_edges
using Random: Random, AbstractRNG
using SimpleTraits: SimpleTraits, @traitfn

# Build an `ITensorNetwork` on the graph backing the inds network `s` from a
# user-supplied per-vertex tensor builder. Each vertex `v` is given a tensor
# whose indices are `s[v]` (the site indices) followed by one fresh `Index` of
# dimension `link_space` for each edge of the graph incident to `v`. The shared
# link index is used on both endpoints so the resulting network's edges agree
# with `edges(s)`.
function _random_tensornetwork(s::IndsNetwork, build_tensor; link_space = 1)
    l = Dict(e => Index(link_space, "Link") for e in edges(s))
    l = merge(l, Dict(reverse(e) => l[e] for e in edges(s)))
    vd = vertex_data(s)
    ts = Dict{eltype(vertices(s)), ITensor}()
    for v in vertices(s)
        site_inds = isassigned(vd, v) ? vd[v] : Index[]
        is = (site_inds..., (l[e] for e in incident_edges(s, v))...)
        ts[v] = build_tensor(is)
    end
    return ITensorNetwork(ts)
end

# Build an ITensor network on a graph specified by the inds network `s`.
# `link_space` sets the bond dimension. Entries are drawn from a standard
# normal distribution.
function random_tensornetwork(
        rng::AbstractRNG, eltype::Type, s::IndsNetwork; kwargs...
    )
    return _random_tensornetwork(
        s, is -> itensor(randn(rng, eltype, dim.(is)...), is...); kwargs...
    )
end

function random_tensornetwork(eltype::Type, s::IndsNetwork; kwargs...)
    return random_tensornetwork(Random.default_rng(), eltype, s; kwargs...)
end

function random_tensornetwork(rng::AbstractRNG, s::IndsNetwork; kwargs...)
    return random_tensornetwork(rng, Float64, s; kwargs...)
end

function random_tensornetwork(s::IndsNetwork; kwargs...)
    return random_tensornetwork(Random.default_rng(), s; kwargs...)
end

@traitfn function random_tensornetwork(
        rng::AbstractRNG, eltype::Type, g::::IsUnderlyingGraph; kwargs...
    )
    return random_tensornetwork(rng, eltype, IndsNetwork(g); kwargs...)
end

@traitfn function random_tensornetwork(eltype::Type, g::::IsUnderlyingGraph; kwargs...)
    return random_tensornetwork(Random.default_rng(), eltype, g; kwargs...)
end

@traitfn function random_tensornetwork(rng::AbstractRNG, g::::IsUnderlyingGraph; kwargs...)
    return random_tensornetwork(rng, Float64, g; kwargs...)
end

@traitfn function random_tensornetwork(g::::IsUnderlyingGraph; kwargs...)
    return random_tensornetwork(Random.default_rng(), g; kwargs...)
end

# Build an ITensor network on a graph specified by the inds network `s`.
# Entries are drawn from the supplied `Distribution`.
function random_tensornetwork(
        rng::AbstractRNG, distribution::Distribution, s::IndsNetwork; kwargs...
    )
    return _random_tensornetwork(
        s, is -> itensor(rand(rng, distribution, dim.(is)...), is...); kwargs...
    )
end

function random_tensornetwork(distribution::Distribution, s::IndsNetwork; kwargs...)
    return random_tensornetwork(Random.default_rng(), distribution, s; kwargs...)
end

@traitfn function random_tensornetwork(
        rng::AbstractRNG, distribution::Distribution, g::::IsUnderlyingGraph; kwargs...
    )
    return random_tensornetwork(rng, distribution, IndsNetwork(g); kwargs...)
end

@traitfn function random_tensornetwork(
        distribution::Distribution, g::::IsUnderlyingGraph; kwargs...
    )
    return random_tensornetwork(Random.default_rng(), distribution, g; kwargs...)
end

function random_ttn(args...; kwargs...)
    return normalize(
        ITensorNetworks._TreeTensorNetwork(random_tensornetwork(args...; kwargs...))
    )
end

function random_mps(args...; kwargs...)
    return random_ttn(args...; kwargs...)
end

function random_mps(f, is::Vector{<:Index}; kwargs...)
    return random_mps(f, ITensorNetworks.path_indsnetwork(is); kwargs...)
end

function random_mps(s::Vector{<:Index}; kwargs...)
    return random_mps(ITensorNetworks.path_indsnetwork(s); kwargs...)
end

# Small grab-bag of model-Hamiltonian builders used in regression tests. Kept
# in a submodule so call sites remain `ModelHamiltonians.ising(g; h = ...)` etc.
module ModelHamiltonians
    using Dictionaries: AbstractDictionary
    using Graphs: AbstractGraph, dst, edges, edgetype, neighborhood, src, vertices
    using ITensors.Ops: OpSum

    to_callable(value::Type) = value
    to_callable(value::Function) = value
    to_callable(value::AbstractDict) = Base.Fix1(getindex, value)
    to_callable(value::AbstractDictionary) = Base.Fix1(getindex, value)
    function to_callable(value::AbstractArray{<:Any, N}) where {N}
        getindex_value(x::Integer) = value[x]
        getindex_value(x::Tuple{Vararg{Integer, N}}) = value[x...]
        getindex_value(x::CartesianIndex{N}) = value[x]
        return getindex_value
    end
    to_callable(value) = Returns(value)

    function nth_nearest_neighbors(g, v, n::Int)
        isone(n) && return neighborhood(g, v, 1)
        return setdiff(neighborhood(g, v, n), neighborhood(g, v, n - 1))
    end

    next_nearest_neighbors(g, v) = nth_nearest_neighbors(g, v, 2)

    function tight_binding(g::AbstractGraph; t = 1, tp = 0, h = 0)
        (; t, tp, h) = map(to_callable, (; t, tp, h))
        h = to_callable(h)
        ℋ = OpSum()
        for e in edges(g)
            ℋ -= t(e), "Cdag", src(e), "C", dst(e)
            ℋ -= t(e), "Cdag", dst(e), "C", src(e)
        end
        for v in vertices(g)
            for nn in next_nearest_neighbors(g, v)
                e = edgetype(g)(v, nn)
                ℋ -= tp(e), "Cdag", src(e), "C", dst(e)
                ℋ -= tp(e), "Cdag", dst(e), "C", src(e)
            end
        end
        for v in vertices(g)
            ℋ -= h(v), "N", v
        end
        return ℋ
    end

    # J1-J2 Heisenberg model on a general graph
    function heisenberg(g::AbstractGraph; J1 = 1, J2 = 0, h = 0)
        (; J1, J2, h) = map(to_callable, (; J1, J2, h))
        ℋ = OpSum()
        for e in edges(g)
            ℋ += J1(e) / 2, "S+", src(e), "S-", dst(e)
            ℋ += J1(e) / 2, "S-", src(e), "S+", dst(e)
            ℋ += J1(e), "Sz", src(e), "Sz", dst(e)
        end
        for v in vertices(g)
            for nn in next_nearest_neighbors(g, v)
                e = edgetype(g)(v, nn)
                ℋ += J2(e) / 2, "S+", src(e), "S-", dst(e)
                ℋ += J2(e) / 2, "S-", src(e), "S+", dst(e)
                ℋ += J2(e), "Sz", src(e), "Sz", dst(e)
            end
        end
        for v in vertices(g)
            ℋ += h(v), "Sz", v
        end
        return ℋ
    end

    # Next-to-nearest-neighbor Ising model (ZZX) on a general graph
    function ising(g::AbstractGraph; J1 = -1, J2 = 0, h = 0)
        (; J1, J2, h) = map(to_callable, (; J1, J2, h))
        ℋ = OpSum()
        for e in edges(g)
            ℋ += J1(e), "Sz", src(e), "Sz", dst(e)
        end
        for v in vertices(g)
            for nn in next_nearest_neighbors(g, v)
                e = edgetype(g)(v, nn)
                if !iszero(J2(e))
                    ℋ += J2(e), "Sz", src(e), "Sz", dst(e)
                end
            end
        end
        for v in vertices(g)
            ℋ += h(v), "Sx", v
        end
        return ℋ
    end

end  # module ModelHamiltonians
