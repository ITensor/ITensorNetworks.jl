using DataGraphs: DataGraphs, set_vertex_data!, underlying_graph, vertex_data
using Graphs: Graphs, AbstractGraph, add_edge!, add_vertex!, edges, has_edge, neighbors,
    rem_edge!, rem_vertex!, vertices
using ITensors: ITensors, ITensor, Index, inds
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype

"""
    ITensorNetwork{V, S}

A tensor network where each vertex holds an `ITensor`. Storage is split
across three fields:

  - `graph::NamedGraph{V}` — the network's graph (`V` is the vertex type),
  - `vertex_data::Dict{V, ITensor}` — the tensor at each vertex,
  - `ind_to_vertices::Dict{Index{S}, Set{V}}` — reverse map from each
    `Index` to the vertices it appears in (`S` is the `Index` space type,
    e.g. `Int` for plain dims or `Vector{Pair{QN, Int}}` for QN-graded).

The reverse map keeps vertex lookup by shared `Index` O(1) and enforces
the graph-edge ↔ shared-`Index` invariant: every `Index` appears at
either one vertex (an external / site index) or two (a bond), and every
graph edge corresponds to exactly the pair of vertices sharing at least
one `Index`. Hyperedges (an `Index` shared by three or more vertices)
are rejected.

# Construction

```julia
ITensorNetwork(tensors)
ITensorNetwork{V}(tensors)
ITensorNetwork{V, S}(tensors)
```

`tensors` is any collection where `keys(tensors)` are vertex labels and
`values(tensors)` are the `ITensor`s at those vertices (e.g. a `Dict`, a
`Dictionary`, or a `Vector{ITensor}` with linear-index vertex labels).
Edges are inferred from shared `Index`es.

# Example

```jldoctest
julia> using ITensors: Index, ITensor

julia> i, j, k = Index(2, "i"), Index(2, "j"), Index(2, "k");

julia> tn = ITensorNetwork([ITensor(i, j), ITensor(j, k)]);

```

See also: `IndsNetwork`, [`TreeTensorNetwork`](@ref ITensorNetworks.TreeTensorNetwork).
"""
struct ITensorNetwork{V, S} <: AbstractITensorNetwork{V}
    graph::NamedGraph{V}
    vertex_data::Dict{V, ITensor}
    ind_to_vertices::Dict{Index{S}, Set{V}}
end

#
# AbstractITensorNetwork interface (field access)
#

DataGraphs.underlying_graph(tn::ITensorNetwork) = tn.graph
DataGraphs.vertex_data(tn::ITensorNetwork) = tn.vertex_data

function DataGraphs.underlying_graph_type(::Type{<:ITensorNetwork{V}}) where {V}
    return NamedGraph{V}
end

#
# Constructors
#

# Infer `S` from a tensor's indices; default to `Int` (plain non-QN
# dim-as-`Int` Indices) when the collection is empty or every tensor
# has no indices.
_index_space_type(::Index{S}) where {S} = S
function _index_space_type(tensors)
    for t in values(tensors)
        is = inds(t)
        isempty(is) || return _index_space_type(first(is))
    end
    return Int
end

# Build the reverse index map from `tensors`, infer the graph edges from
# that map, and enforce the no-hyperedge invariant.
function ITensorNetwork{V, S}(tensors) where {V, S}
    vs = V[v for v in keys(tensors)]
    graph = NamedGraph(vs)
    vertex_data = Dict{V, ITensor}(v => tensors[v] for v in vs)
    ind_to_vertices = Dict{Index{S}, Set{V}}()
    for v in vs, i in inds(vertex_data[v])
        push!(get!(ind_to_vertices, i, Set{V}()), v)
    end
    for (i, owners) in ind_to_vertices
        length(owners) <= 2 || error(
            "Index $i appears at $(length(owners)) vertices; `ITensorNetwork` " *
                "is not a hypergraph — every `Index` must appear at one (external) " *
                "or two (bond) vertices."
        )
    end
    # Walk `vs` in order so the edge add order — and therefore the
    # `neighbors(g, v)` / `edges(g)` iteration order — is deterministic in
    # the input vertex order, rather than the non-deterministic hash order
    # of `values(ind_to_vertices)` and `Set` iteration.
    for v in vs, i in inds(vertex_data[v])
        for u in ind_to_vertices[i]
            u != v && !has_edge(graph, v, u) && add_edge!(graph, v => u)
        end
    end
    return ITensorNetwork{V, S}(graph, vertex_data, ind_to_vertices)
end

function ITensorNetwork{V, S}(tn::AbstractITensorNetwork) where {V, S}
    return ITensorNetwork{V, S}(Dict{V, ITensor}(v => tn[v] for v in vertices(tn)))
end

function ITensorNetwork{V}(tensors) where {V}
    return ITensorNetwork{V, _index_space_type(tensors)}(tensors)
end

function ITensorNetwork{V}(tn::AbstractITensorNetwork) where {V}
    return ITensorNetwork{V}(Dict{V, ITensor}(v => tn[v] for v in vertices(tn)))
end

ITensorNetwork(tensors) = ITensorNetwork{keytype(tensors)}(tensors)

# Empty network over `vertices(g)`: vertices are added to the graph but
# carry no tensor data yet, and the graph has no edges. Tensors are
# populated via `setindex!`, which infers bonds from shared `Index`es;
# any edges already in `g` are discarded since they'll be re-derived from
# the indices. Primarily used by `similar_graph` so that `induced_subgraph`
# and related operations can build their result incrementally.
function ITensorNetwork{V, S}(g::AbstractGraph) where {V, S}
    return ITensorNetwork{V, S}(
        NamedGraph(collect(V, vertices(g))),
        Dict{V, ITensor}(),
        Dict{Index{S}, Set{V}}()
    )
end

#
# Vertex-type conversion and copy
#

NamedGraphs.convert_vertextype(::Type{V}, tn::ITensorNetwork{V}) where {V} = tn
function NamedGraphs.convert_vertextype(::Type{V}, tn::ITensorNetwork) where {V}
    return ITensorNetwork{V}(Dict{V, ITensor}(v => tn[v] for v in vertices(tn)))
end

function Base.copy(tn::ITensorNetwork{V, S}) where {V, S}
    return ITensorNetwork{V, S}(
        Dict{V, ITensor}(v => copy(tn[v]) for v in vertices(tn))
    )
end

#
# Mutation: keep `graph`, `vertex_data`, and `ind_to_vertices` in sync.
#

# Write `value` to vertex `v`, updating the reverse map and reconciling
# edges so the graph-edge ↔ shared-`Index` invariant holds. Cost is
# O(deg(v) + |inds(value)|).
function DataGraphs.set_vertex_data!(tn::ITensorNetwork{V, S}, value, v) where {V, S}
    # Unregister old inds of `tn[v]` from the reverse map.
    if haskey(tn.vertex_data, v)
        for i in inds(tn.vertex_data[v])
            owners = tn.ind_to_vertices[i]
            delete!(owners, v)
            isempty(owners) && delete!(tn.ind_to_vertices, i)
        end
    end
    # Write the new tensor.
    tn.vertex_data[v] = value
    # Register new inds.
    for i in inds(value)
        push!(get!(tn.ind_to_vertices, i, Set{V}()), v)
        length(tn.ind_to_vertices[i]) <= 2 || error(
            "Index $i now appears at $(length(tn.ind_to_vertices[i])) vertices; " *
                "`ITensorNetwork` forbids hyperedges (3+ vertices sharing an `Index`)."
        )
    end
    # Reconcile graph edges incident to `v` against the reverse map.
    desired = Set{V}()
    for i in inds(value)
        for u in tn.ind_to_vertices[i]
            u == v || push!(desired, u)
        end
    end
    for u in collect(neighbors(tn.graph, v))
        u in desired || rem_edge!(tn.graph, v => u)
    end
    for u in desired
        has_edge(tn.graph, v, u) || add_edge!(tn.graph, v => u)
    end
    return tn
end

# Drop `v` from the reverse map, vertex data, and graph in one shot.
function Graphs.rem_vertex!(tn::ITensorNetwork, v)
    if haskey(tn.vertex_data, v)
        for i in inds(tn.vertex_data[v])
            owners = tn.ind_to_vertices[i]
            delete!(owners, v)
            isempty(owners) && delete!(tn.ind_to_vertices, i)
        end
        delete!(tn.vertex_data, v)
    end
    rem_vertex!(tn.graph, v)
    return tn
end

# Add `v` to the graph without any tensor data. A subsequent
# `tn[v] = tensor` writes the tensor and reconciles edges.
function Graphs.add_vertex!(tn::ITensorNetwork, v)
    add_vertex!(tn.graph, v)
    return tn
end

# Fresh `ITensorNetwork` over `vertices(g)` with no tensors. Used by
# `induced_subgraph_from_vertices` to build a same-typed empty container
# that subsequent `setindex!` calls populate.
function NamedGraphs.similar_graph(tn::ITensorNetwork{V, S}, g::AbstractGraph) where {V, S}
    return ITensorNetwork{V, S}(g)
end
