using DataGraphs:
    DataGraphs, DataGraph, is_vertex_assigned, set_vertex_data!, underlying_graph
using Graphs: Graphs, add_edge!, edgetype, has_edge, neighbors, rem_edge!, rem_vertex!
using ITensors: ITensors, ITensor, Index, inds
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, similar_graph, vertextype

"""
    ITensorNetwork{V}

A tensor network where each vertex holds an `ITensor`. The network graph is a
`NamedGraph{V}` and edges represent shared indices between neighboring tensors.

The type maintains a reverse index map (`Index → vertices`) so that vertex
lookup by shared `Index` is O(1) and the graph-edge ↔ shared-`Index`
correspondence is reconciled in O(deg(v) + |inds(tn[v])|) on every tensor
write.

# Constructors

**From a collection of `ITensor`s** (edges inferred from shared indices):

```julia
ITensorNetwork(tensors)
```

`tensors` is any collection where `keys(tensors)` are vertex labels and
`values(tensors)` are the `ITensor`s at those vertices (e.g. a `Dict`, a
`Dictionary`, or a `Vector{ITensor}` with linear-index vertex labels).

**From a collection of `ITensor`s placed at the vertices of a given graph**
(no edge inference; the caller is responsible for the edges):

```julia
ITensorNetwork(tensors, graph::NamedGraph)
```

# Example

```jldoctest
julia> using ITensors: Index, ITensor

julia> i, j, k = Index(2, "i"), Index(2, "j"), Index(2, "k");

julia> tn = ITensorNetwork([ITensor(i, j), ITensor(j, k)]);

```

See also: `IndsNetwork`, [`TreeTensorNetwork`](@ref ITensorNetworks.TreeTensorNetwork).
"""
struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
    data_graph::DataGraph{V, ITensor, ITensor, NamedGraph{V}, NamedEdge{V}}
    # Reverse index map: for every `Index` appearing in any vertex tensor,
    # the set of vertices holding that `Index`. Maintained by `setindex!`
    # so that edge reconciliation after a write is O(deg(v) + |inds(tn[v])|)
    # instead of an O(n) sweep over all vertices.
    ind_to_vertices::Dict{Index, Set{V}}

    # Sole inner ctor: place `tensors` at the vertices of `graph` and build
    # the reverse map from the resulting tensors. The graph's edges are taken
    # at face value; callers are responsible for the graph-edge ↔
    # shared-`Index` invariant on construction (the public ctors below do
    # this either by trusting the caller's graph or by re-inferring edges).
    function ITensorNetwork{V}(tensors, graph::NamedGraph) where {V}
        g = NamedGraph{V}(graph)
        dg = DataGraph(g; vertex_data_type = ITensor, edge_data_type = ITensor)
        for v in vertices(g)
            dg[v] = tensors[v]
        end
        ind_to_vertices = Dict{Index, Set{V}}()
        for v in vertices(dg)
            for i in inds(dg[v])
                push!(get!(ind_to_vertices, i, Set{V}()), v)
            end
        end
        return new{V}(dg, ind_to_vertices)
    end
end

#
# Data access
#

data_graph(tn::ITensorNetwork) = getfield(tn, :data_graph)
data_graph_type(TN::Type{<:ITensorNetwork}) = fieldtype(TN, :data_graph)

function DataGraphs.underlying_graph_type(TN::Type{<:ITensorNetwork})
    return fieldtype(data_graph_type(TN), :underlying_graph)
end

#
# Reverse index map and edge reconciliation (implementation detail)
#

# Internal accessor — keep `ind_to_vertices` package-private so that the
# `AbstractITensorNetwork` interface stays oblivious to the reverse map.
_ind_to_vertices(tn::ITensorNetwork) = getfield(tn, :ind_to_vertices)

# Write `value` to `v` and re-establish the graph-edge ↔ shared-`Index`
# invariant: incident edges of `v` are exactly the vertices sharing an
# `Index` with `value`. The reverse map makes the reconciliation
# O(deg(v) + |inds(value)|), so `setindex!` does it unconditionally and
# the old `@preserve_graph` / `fix_edges!` bypass is no longer needed.
function DataGraphs.set_vertex_data!(tn::ITensorNetwork, value, v)
    _unregister_inds!(tn, v)
    data_graph(tn)[v] = value
    _register_inds!(tn, v)
    _reconcile_edges!(tn, v)
    return tn
end

# Drop `v` from the reverse map entry of each `Index` currently in `tn[v]`.
function _unregister_inds!(tn::ITensorNetwork, v)
    is_vertex_assigned(tn, v) || return tn
    map = _ind_to_vertices(tn)
    for i in inds(tn[v])
        haskey(map, i) || continue
        vs = map[i]
        delete!(vs, v)
        isempty(vs) && delete!(map, i)
    end
    return tn
end

# Mirror vertex removal on the reverse map. `contract`, `induced_subgraph`,
# etc. structurally edit the graph and would otherwise leave stale entries
# behind, causing later edge reconciliation to point at vertices that no
# longer exist. Routes the underlying-graph update through the
# `AbstractDataGraph` fallback (which only touches the graph structure)
# instead of `DataGraph`'s override (which insists on deleting edge_data
# for every incident edge — `ITensorNetwork` edges carry no edge_data).
function Graphs.rem_vertex!(tn::ITensorNetwork, v)
    _unregister_inds!(tn, v)
    rem_vertex!(underlying_graph(data_graph(tn)), v)
    return tn
end

# Register `v` against each `Index` in `tn[v]`.
function _register_inds!(tn::ITensorNetwork{V}, v) where {V}
    map = _ind_to_vertices(tn)
    for i in inds(tn[v])
        push!(get!(map, i, Set{V}()), v)
    end
    return tn
end

# Reconcile the graph edges incident to `v` so that they match exactly the
# set of vertices sharing an `Index` with `tn[v]`. O(deg(v) + |inds(tn[v])|).
function _reconcile_edges!(tn::ITensorNetwork{V}, v) where {V}
    map = _ind_to_vertices(tn)
    desired = Set{V}()
    for i in inds(tn[v])
        for u in map[i]
            u == v || push!(desired, u)
        end
    end
    # `DataGraphs.rem_edge!` requires edge_data to be assigned for the edge
    # — but `ITensorNetwork` edges carry no edge_data, so bypass it and
    # work directly on the underlying `NamedGraph`. Edge inserts can stay
    # on the `DataGraph` since `add_edge!` doesn't touch edge_data.
    dg = data_graph(tn)
    ug = underlying_graph(dg)
    E = edgetype(tn)
    for u in collect(neighbors(tn, v))
        u in desired || rem_edge!(ug, E(v, u))
    end
    for u in desired
        has_edge(tn, E(v, u)) || add_edge!(dg, E(v, u))
    end
    return tn
end

#
# Construction from collections of ITensors
#

# Tensors only: derive the vertex list from `keys(tensors)`. Build an empty
# network on that vertex set, then write each tensor via `setindex!`; the
# reverse-index map drives edge reconciliation as each tensor lands, so edges
# are inferred in O(sum_v |inds(tn[v])|) total rather than an O(n²) sweep.
function ITensorNetwork{V}(tensors) where {V}
    # Build the vertex list with element type `V` so that an empty `tensors`
    # input doesn't get the graph's vertex type inferred to whatever
    # `keys(tensors)` happens to give (e.g. `Int` for an empty `Vector{ITensor}`).
    g = NamedGraph(V[v for v in keys(tensors)])
    default = Dict{V, ITensor}(v => ITensor() for v in vertices(g))
    tn = ITensorNetwork(default, g)
    for v in vertices(g)
        tn[v] = tensors[v]
    end
    return tn
end

# Non-parametric delegates: extract `V` via `keytype` / `vertextype`.
function ITensorNetwork(tensors)
    return ITensorNetwork{keytype(tensors)}(tensors)
end
function ITensorNetwork(tensors, graph::NamedGraph)
    return ITensorNetwork{vertextype(graph)}(tensors, graph)
end

#
# Vertex-type conversion and copy
#

function ITensorNetwork{V}(tn::ITensorNetwork) where {V}
    g = NamedGraph{V}(underlying_graph(tn))
    tensors = Dict{V, ITensor}(v => tn[v] for v in vertices(tn))
    return ITensorNetwork(tensors, g)
end

ITensorNetwork(tn::ITensorNetwork) = copy(tn)

NamedGraphs.convert_vertextype(::Type{V}, tn::ITensorNetwork{V}) where {V} = tn
NamedGraphs.convert_vertextype(V::Type, tn::ITensorNetwork) = ITensorNetwork{V}(tn)

function Base.copy(tn::ITensorNetwork{V}) where {V}
    g = copy(underlying_graph(tn))
    tensors = Dict{V, ITensor}(v => copy(tn[v]) for v in vertices(g))
    return ITensorNetwork(tensors, g)
end

function NamedGraphs.similar_graph(tn::ITensorNetwork, underlying_graph::AbstractGraph)
    g = NamedGraph(underlying_graph)
    default = Dict{vertextype(g), ITensor}(v => ITensor() for v in vertices(g))
    return ITensorNetwork(default, g)
end
