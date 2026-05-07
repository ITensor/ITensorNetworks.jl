using DataGraphs: DataGraphs, DataGraph
using ITensors: ITensors, ITensor
using NamedGraphs: NamedGraphs, NamedEdge, NamedGraph, similar_graph, vertextype

"""
    ITensorNetwork{V}

A tensor network where each vertex holds an `ITensor`. The network graph is a
`NamedGraph{V}` and edges represent shared indices between neighboring tensors.

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

See also: `IndsNetwork`, [`ttn`](@ref ITensorNetworks.ttn), [`TreeTensorNetwork`](@ref ITensorNetworks.TreeTensorNetwork).
"""
const _ITensorCollection = Union{
    AbstractVector{<:ITensor},
    AbstractDict{<:Any, <:ITensor},
    AbstractDictionary{<:Any, <:ITensor},
}

struct ITensorNetwork{V} <: AbstractITensorNetwork{V}
    data_graph::DataGraph{V, ITensor, ITensor, NamedGraph{V}, NamedEdge{V}}

    # Sole inner ctor: place `tensors` at the vertices of `graph`. No checks —
    # `tensors` must be indexable at every vertex, the graph's edges are
    # taken at face value.
    function ITensorNetwork{V}(
            tensors::_ITensorCollection, graph::NamedGraph
        ) where {V}
        g = NamedGraph{V}(graph)
        dg = DataGraph(g; vertex_data_type = ITensor, edge_data_type = ITensor)
        for v in vertices(g)
            dg[v] = tensors[v]
        end
        return new{V}(dg)
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
# Construction from collections of ITensors
#

# Tensors only: derive graph from `keys(tensors)`, then run edge inference.
# Without the reverse index map, edge inference is O(n²).
function ITensorNetwork{V}(tensors::_ITensorCollection) where {V}
    # Build the vertex list with element type `V` so that an empty `tensors`
    # input doesn't get the graph's vertex type inferred to whatever
    # `keys(tensors)` happens to give (e.g. `Int` for an empty `Vector{ITensor}`).
    g = NamedGraph(V[v for v in keys(tensors)])
    # Annotate the default Dict's element type explicitly: when the
    # comprehension is empty (e.g. constructing an empty `ITensorNetwork{V}`
    # via the bootstrap call inside the IndsNetwork function-callback ctor)
    # type inference can give `Dict{Any, Any}` for `V = Any`, which then
    # falls outside `_ITensorCollection` and routes the next dispatch
    # through the wrong path (function-callback Group C) — infinite
    # recursion. Locking the value type to `ITensor` avoids that.
    default = Dict{V, ITensor}(v => ITensor() for v in vertices(g))
    tn = ITensorNetwork(default, g)
    for v in vertices(g)
        tn[v] = tensors[v]
    end
    return tn
end

# Non-parametric delegates: extract `V` via `keytype` / `vertextype`.
function ITensorNetwork(tensors::_ITensorCollection)
    return ITensorNetwork{keytype(tensors)}(tensors)
end
function ITensorNetwork(tensors::_ITensorCollection, graph::NamedGraph)
    return ITensorNetwork{vertextype(graph)}(tensors, graph)
end

#
# Vertex-type conversion and copy
#

function ITensorNetwork{V}(tn::ITensorNetwork) where {V}
    g = NamedGraph{V}(underlying_graph(tn))
    # Type-annotated so empty-`tn` + `V = Any` doesn't drop us out of
    # `_ITensorCollection` and into the wrong dispatch (see the parametric
    # `(tensors)` form above for the same gotcha).
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
