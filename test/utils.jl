# Test-only helpers extracted from `src/`. Lives at `test/utils.jl` so the
# `ITensorPkgSkeleton.runtests` runner doesn't auto-pick it up (it only globs
# `test_*.jl`). Each test file that needs these does `include("utils.jl")`
# inside its gensym module.

using DataGraphs: underlying_graph, vertex_data
using Dictionaries: AbstractDictionary
using Graphs: AbstractGraph, edges, vertices
using ITensorNetworks: ITensorNetwork, IndsNetwork, insert_linkinds
using ITensors.NDTensors: dim
using ITensors: ITensors, ITensor, Index, contract, itensor, onehot
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs: NamedGraph
using Random: Random, AbstractRNG

# --- random_tensornetwork ----------------------------------------------------

# At each vertex of `s`'s graph, place an `itensor(randn(rng, eltype, ...), inds_v)`
# whose inds are the site inds at that vertex (from `s[v]`, or empty if unassigned)
# concatenated with one fresh `Index(link_space, "Link")` per incident edge, shared
# with the other endpoint.
function random_tensornetwork(
        rng::AbstractRNG, eltype::Type, s::IndsNetwork; link_space = 1
    )
    g = NamedGraph(underlying_graph(s))
    links = Dict(e => Index(link_space, "Link") for e in edges(g))
    tensors = Dict(
        map(collect(vertices(g))) do v
            site_v = isassigned(vertex_data(s), v) ? s[v] : Index[]
            link_v = [
                haskey(links, e) ? links[e] : links[reverse(e)]
                    for e in incident_edges(g, v)
            ]
            inds_v = [site_v; link_v]
            return v => itensor(randn(rng, eltype, dim.(inds_v)...), inds_v)
        end
    )
    return ITensorNetwork(tensors, g)
end
function random_tensornetwork(
        rng::AbstractRNG, eltype::Type, g::AbstractGraph; kwargs...
    )
    return random_tensornetwork(rng, eltype, IndsNetwork(g); kwargs...)
end

# RNG / eltype / both defaults
function random_tensornetwork(rng::AbstractRNG, sites_or_graph; kwargs...)
    return random_tensornetwork(rng, Float64, sites_or_graph; kwargs...)
end
function random_tensornetwork(eltype::Type, sites_or_graph; kwargs...)
    return random_tensornetwork(Random.default_rng(), eltype, sites_or_graph; kwargs...)
end
function random_tensornetwork(sites_or_graph; kwargs...)
    return random_tensornetwork(Random.default_rng(), Float64, sites_or_graph; kwargs...)
end

# --- productstate -------------------------------------------------------------

# Convert a state spec (string / dict / dictionary / array / function) into a
# callable mapping vertex -> state-spec-for-that-vertex.
_to_callable(value::Function) = value
_to_callable(value::AbstractDict) = Base.Fix1(getindex, value) ∘ keytype(value)
_to_callable(value::AbstractDictionary) = Base.Fix1(getindex, value) ∘ keytype(value)
_to_callable(value::AbstractArray) = Base.Fix1(getindex, value) ∘ CartesianIndex
_to_callable(value) = Returns(value)

# Build a product-state ITensorNetwork: at each vertex, the on-site state
# vector (looked up via `ITensors.state(name, site_index)`) is contracted with
# `onehot` vectors on each incident link Index. Link inds are filled in by
# `insert_linkinds`, which picks the right `trivial_space` (plain `1` or
# `[QN() => 1]`) to match the site indices' QN structure.
function productstate(elt::Type, state, s::IndsNetwork; kwargs...)
    state_at = _to_callable(state)
    s = insert_linkinds(s; kwargs...)
    g = NamedGraph(underlying_graph(s))
    tensors = Dict(
        map(collect(vertices(g))) do v
            site_v = isassigned(vertex_data(s), v) ? s[v] : Index[]
            link_v = reduce(vcat, (s[e] for e in incident_edges(s, v)); init = Index[])
            site_t = ITensors.state(state_at(v), only(site_v))
            t = contract([site_t; (onehot(i => 1) for i in link_v)...])
            return v => ITensors.convert_eltype(elt, t)
        end
    )
    return ITensorNetwork(tensors, g)
end

function productstate(state, s::IndsNetwork; kwargs...)
    return productstate(Float64, state, s; kwargs...)
end

# --- ModelHamiltonians --------------------------------------------------------

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
