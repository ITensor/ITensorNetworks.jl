using DataGraphs: DataGraphs, set_vertex_data!, underlying_graph, vertex_data
using Dictionaries: Dictionaries, Dictionary
using Graphs: Graphs, add_edge!, add_vertex!, edges, has_edge, has_vertex, neighbors,
    rem_edge!, rem_vertex!, vertices
using ITensors: ITensors, ITensor, Index, inds
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, vertextype

"""
    ITensorNetwork{V}

A tensor network where each vertex holds an `ITensor`. Storage is split
across three fields:

  - `graph::NamedGraph{V}` — the network's graph (`V` is the vertex type),
  - `vertex_data::Dictionary{V, ITensor}` — the tensor at each vertex,
  - `ind_to_vertices::Dict{Index, Set{V}}` — reverse map from each `Index`
    to the vertices it appears in.

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
struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
    graph::NamedGraph{V}
    vertex_data::Dictionary{V, ITensor}
    ind_to_vertices::Dict{Index, Set{V}}
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

# Construct by feeding `tensors` through `set_vertex_data!` one vertex
# at a time — this centralizes the reverse-map registration, edge
# inference, and hypergraph check in a single place (the `setindex!`
# code path). Walking `keys(tensors)` in order makes the resulting
# `neighbors(g, v)` / `edges(g)` iteration order deterministic in the
# input order.
function ITensorNetwork{V}(tensors) where {V}
    tn = ITensorNetwork{V}(NamedGraph{V}(), Dictionary{V, ITensor}(), Dict{Index, Set{V}}())
    for v in keys(tensors)
        set_vertex_data!(tn, tensors[v], v)
    end
    return tn
end

ITensorNetwork(tensors) = ITensorNetwork{keytype(tensors)}(tensors)

#
# Vertex-type conversion and copy
#

NamedGraphs.convert_vertextype(::Type{V}, tn::ITensorNetwork{V}) where {V} = tn
function NamedGraphs.convert_vertextype(::Type{V}, tn::ITensorNetwork) where {V}
    return ITensorNetwork{V}(tn)
end

Base.copy(tn::ITensorNetwork) = ITensorNetwork(map(copy, vertex_data(tn)))

#
# Mutation: keep `graph`, `vertex_data`, and `ind_to_vertices` in sync.
#

# Write `value` to vertex `v`, updating the reverse map and reconciling
# edges so the graph-edge ↔ shared-`Index` invariant holds. Cost is
# O(deg(v) + |inds(value)|). If `v` isn't already in the network, it's
# added — so this is also the natural way to grow the network one tensor
# at a time without a separate `add_vertex!` step. Operates on raw
# storage so `ITensorNetwork` and `TreeTensorNetwork` can share it.
function _set_vertex_data!(
        graph::NamedGraph{V},
        vertex_data::Dictionary{V, ITensor},
        ind_to_vertices::Dict{Index, Set{V}},
        value,
        v
    ) where {V}
    # Add the vertex to the graph if it's new.
    has_vertex(graph, v) || add_vertex!(graph, v)
    # Unregister old inds of `vertex_data[v]` from the reverse map.
    if haskey(vertex_data, v)
        for i in inds(vertex_data[v])
            owners = ind_to_vertices[i]
            delete!(owners, v)
            isempty(owners) && delete!(ind_to_vertices, i)
        end
    end
    # Write the new tensor. `Dictionaries.set!` inserts or updates;
    # plain `setindex!` would error on a vertex not already in the dict.
    Dictionaries.set!(vertex_data, v, value)
    # Register new inds.
    for i in inds(value)
        push!(get!(ind_to_vertices, i, Set{V}()), v)
        length(ind_to_vertices[i]) <= 2 || error(
            "Index $i now appears at $(length(ind_to_vertices[i])) vertices; " *
                "`ITensorNetwork` forbids hyperedges (3+ vertices sharing an `Index`)."
        )
    end
    # Reconcile graph edges incident to `v` against the reverse map.
    desired = Set{V}()
    for i in inds(value)
        for u in ind_to_vertices[i]
            u == v || push!(desired, u)
        end
    end
    for u in collect(neighbors(graph, v))
        u in desired || rem_edge!(graph, v => u)
    end
    for u in desired
        has_edge(graph, v, u) || add_edge!(graph, v => u)
    end
    return nothing
end

function DataGraphs.set_vertex_data!(tn::ITensorNetwork, value, v)
    _set_vertex_data!(tn.graph, tn.vertex_data, tn.ind_to_vertices, value, v)
    return tn
end

# Drop `v` from the reverse map, vertex data, and graph in one shot.
function _rem_vertex!(
        graph::NamedGraph{V},
        vertex_data::Dictionary{V, ITensor},
        ind_to_vertices::Dict{Index, Set{V}},
        v
    ) where {V}
    if haskey(vertex_data, v)
        for i in inds(vertex_data[v])
            owners = ind_to_vertices[i]
            delete!(owners, v)
            isempty(owners) && delete!(ind_to_vertices, i)
        end
        delete!(vertex_data, v)
    end
    rem_vertex!(graph, v)
    return nothing
end

function Graphs.rem_vertex!(tn::ITensorNetwork, v)
    _rem_vertex!(tn.graph, tn.vertex_data, tn.ind_to_vertices, v)
    return tn
end
