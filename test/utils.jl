# Test-only helpers extracted from `src/`. Lives at `test/utils.jl` so the
# `ITensorPkgSkeleton.runtests` runner doesn't auto-pick it up (it only globs
# `test_*.jl`). Each test file that needs these does `include("utils.jl")`
# inside its gensym module.

using DataGraphs: underlying_graph, vertex_data
using Graphs: AbstractGraph, add_edge!, dst, edges, src, vertices
using ITensorNetworks: @preserve_graph, ITensorNetwork, IndsNetwork, data_graph
using ITensors.NDTensors: dim
using ITensors: ITensors, ITensor, Index, QN, dag, hasqns, inds, itensor, onehot
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs: NamedGraph
using Random: Random, AbstractRNG

# Pick a length-1 space matching `tn`'s Index style: a dense `1` if no
# vertex tensor has QN-graded indices, otherwise a single-block `[QN() => 1]`.
# Used by `_add_edge!` below to thread a trivial Link Index between two
# tensors without disturbing the QN structure (real `Index` constructors
# need a space, not just a dimension).
function _trivial_link_space(tn)
    any(hasqns ∘ inds, (tn[v] for v in vertices(tn))) || return 1
    return [QN() => 1]
end

# --- random_tensornetwork ----------------------------------------------------

# Core: at each vertex of `graph`, place an `itensor(randn(rng, eltype, ...), inds_v)`
# whose inds are `siteinds[v]` concatenated with one fresh `Index(link_space, "Link")`
# per incident edge, shared with the other endpoint. `siteinds` is anything indexable
# by vertex (`keys(siteinds)` matches `vertices(graph)`); use `Index[]` per vertex for
# no site inds.
function random_tensornetwork(
        rng::AbstractRNG, eltype::Type, graph::AbstractGraph, siteinds; link_space = 1
    )
    g = NamedGraph(graph)
    links = Dict(e => Index(link_space, "Link") for e in edges(g))
    links = merge(links, Dict(reverse(e) => links[e] for e in edges(g)))
    tensors = Dict(
        map(collect(vertices(g))) do v
            link_v = [links[e] for e in incident_edges(g, v)]
            inds_v = [siteinds[v]; link_v]
            return v => itensor(randn(rng, eltype, dim.(inds_v)...), inds_v)
        end
    )
    return ITensorNetwork(tensors, g)
end

# `IndsNetwork`: extract site inds (`Index[]` where unassigned).
function random_tensornetwork(
        rng::AbstractRNG, eltype::Type, s::IndsNetwork; kwargs...
    )
    siteinds = Dict(
        v => isassigned(vertex_data(s), v) ? s[v] : Index[] for v in vertices(s)
    )
    return random_tensornetwork(rng, eltype, underlying_graph(s), siteinds; kwargs...)
end

# Plain graph: no site inds.
function random_tensornetwork(
        rng::AbstractRNG, eltype::Type, g::AbstractGraph; kwargs...
    )
    return random_tensornetwork(
        rng,
        eltype,
        g,
        Dict(v => Index[] for v in vertices(g));
        kwargs...
    )
end

# RNG / eltype / both defaults
function random_tensornetwork(rng::AbstractRNG, sites; kwargs...)
    return random_tensornetwork(rng, Float64, sites; kwargs...)
end
function random_tensornetwork(eltype::Type, sites; kwargs...)
    return random_tensornetwork(Random.default_rng(), eltype, sites; kwargs...)
end
function random_tensornetwork(sites; kwargs...)
    return random_tensornetwork(Random.default_rng(), Float64, sites; kwargs...)
end

# --- productstate -------------------------------------------------------------

# Thread a fresh trivial-QN dim-1 link Index between `src(edge)` and `dst(edge)`
# by outer-producting `onehot` basis vectors into each endpoint tensor. This is
# the productstate-specific analog of `ITensorNetworks.add_edge!` — the latter
# uses QR, which would push site-state QN flux into the link and leave BP's
# `default_message` (single-index `delta`) with no compatible block. `onehot`
# defaults to `Float64`, so we typecast it to `elt` to keep `productstate(elt,
# ...)` element-type-preserving. Uses `@preserve_graph` plus an explicit
# `add_edge!` on the underlying graph so the graph-edge ↔ shared-Index
# invariant is maintained without relying on auto-reconciliation.
function _add_edge!(elt::Type, tn, edge)
    iₑ = Index(_trivial_link_space(tn), "Link")
    X = ITensors.convert_eltype(elt, onehot(iₑ => 1))
    @preserve_graph tn[src(edge)] = tn[src(edge)] * X
    @preserve_graph tn[dst(edge)] = tn[dst(edge)] * dag(X)
    add_edge!(data_graph(tn), edge)
    return tn
end

# Build a product-state ITensorNetwork: start from a site-only TN (one tensor
# per vertex with just the on-site state vector, looked up via
# `ITensors.state(name, site_index)`), then add each edge from the original
# IndsNetwork via `_add_edge!`. `state` is anything indexable by vertex (dict,
# dictionary, array, ...); the `Function` method just materializes a Dict first.
function productstate(elt::Type, state::Function, s::IndsNetwork)
    return productstate(elt, Dict(v => state(v) for v in vertices(s)), s)
end
function productstate(elt::Type, state, s::IndsNetwork)
    g = NamedGraph(collect(vertices(s)))
    tensors = Dict(
        map(collect(vertices(s))) do v
            site_v = isassigned(vertex_data(s), v) ? s[v] : Index[]
            t = ITensors.state(state[v], only(site_v))
            return v => ITensors.convert_eltype(elt, t)
        end
    )
    tn = ITensorNetwork(tensors, g)
    for e in edges(s)
        _add_edge!(elt, tn, e)
    end
    return tn
end
productstate(state, s::IndsNetwork) = productstate(Float64, state, s)

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

# --- OpSum helpers ------------------------------------------------------------

# Vertex-relabeling for OpSum trees: walks `Sum`/`Prod`/`Scaled`/`Op` nodes and
# applies `f` to every site label inside `Op`. The `Scaled` step also runs the
# coefficient through `maybe_real` to drop spurious complex zero imaginary
# parts. Used by `test_opsum_to_ttn_mpo_cross_check.jl` to compare against
# `ITensorMPS.MPO` by reindexing graph vertices onto an MPS line.
using ITensorNetworks: maybe_real
using ITensors.LazyApply: Prod, Scaled, Sum
using ITensors.Ops: Op, Ops

# Promote a scalar / tuple-of-scalars into tuple form. Used below by
# `group_terms` to compare each `OpSum` term's site list against `[src(e),
# dst(e)]` uniformly, regardless of whether sites are bare scalars or already
# tuples (NamedGraph vertex labels can be either).
to_tuple(x) = (x,)
to_tuple(x::Tuple) = x

function replace_vertices(f, ∑o::Sum)
    return Sum(map(oᵢ -> replace_vertices(f, oᵢ), Ops.terms(∑o)))
end

function replace_vertices(f, ∏o::Prod)
    return Prod(map(oᵢ -> replace_vertices(f, oᵢ), Ops.terms(∏o)))
end

function replace_vertices(f, o::Scaled)
    return maybe_real(Ops.coefficient(o)) * replace_vertices(f, Ops.argument(o))
end

set_sites(o::Op, sites) = Op(Ops.which_op(o), sites...; Ops.params(o)...)

function replace_vertices(f, o::Op)
    return set_sites(o, f.(Ops.sites(o)))
end

# Bucket `ITensors.terms(ℋ)` by which edge of `g` they sit on. Each term that
# lives on a single edge `e = src => dst` gets summed into one bucket; the
# returned `Sum` carries those per-edge sums in the order `edges(g)` produces
# them. Used by `test_tebd.jl` to pre-group an Ising Hamiltonian before feeding
# it to TEBD, which expects per-edge terms.
using SplitApplyCombine: group

function group_terms(ℋ::Sum, g)
    grouped_terms = group(ITensors.terms(ℋ)) do t
        findfirst(edges(g)) do e
            return to_tuple.(ITensors.sites(t)) ⊆ [src(e), dst(e)]
        end
    end
    return Sum(collect(sum.(grouped_terms)))
end
