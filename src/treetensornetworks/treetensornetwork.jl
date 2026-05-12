using DataGraphs: DataGraphs, set_vertex_data!, underlying_graph, vertex_data
using Dictionaries: Dictionary, Indices
using Graphs: Graphs, is_tree, rem_vertex!, vertices
using ITensors: ITensor, Index
using NamedGraphs.GraphsExtensions: vertextype
using NamedGraphs: NamedGraph

"""
    TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}

A tensor network whose underlying graph is a tree. Storage mirrors
[`ITensorNetwork`](@ref) — `graph`, `vertex_data`, and `ind_to_vertices`
— plus an `ortho_region` tracking the orthogonality center.

`TTN` is an alias for `TreeTensorNetwork`.

Use the [`TreeTensorNetwork`](@ref) constructors to build instances, and
[`orthogonalize`](@ref) to bring the network into a canonical gauge.

See also: [`ITensorNetwork`](@ref).
"""
struct TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}
    graph::NamedGraph{V}
    vertex_data::Dictionary{V, ITensor}
    ind_to_vertices::Dict{Index, Set{V}}
    ortho_region::Indices{V}
end

# Empty TTN with no vertices. The is-a-tree invariant holds trivially.
function TreeTensorNetwork{V}() where {V}
    return TreeTensorNetwork{V}(
        NamedGraph{V}(),
        Dictionary{V, ITensor}(),
        Dict{Index, Set{V}}(),
        Indices{V}()
    )
end
TreeTensorNetwork() = TreeTensorNetwork{Any}()

"""
    TreeTensorNetwork(tensors) -> TreeTensorNetwork

Construct a `TreeTensorNetwork` from any collection of tensors accepted by
`ITensorNetwork` (e.g. a `Dict`, `Dictionary`, a `Vector{ITensor}`, or another
`AbstractITensorNetwork`). Edges are inferred from shared `Index`es; the
underlying graph must be a tree.

The result starts with `ortho_region == vertices(tn)` — i.e. no particular
gauge is assumed. Use [`orthogonalize`](@ref) to bring the network into a
canonical gauge centered at a chosen vertex or region.

# Example

```jldoctest
julia> using ITensors: Index, ITensor

julia> i, j, k = Index(2, "i"), Index(2, "j"), Index(2, "k");

julia> ttn = TreeTensorNetwork([ITensor(i, j), ITensor(j, k)]);

```

See also: [`ITensorNetwork`](@ref), [`orthogonalize`](@ref).
"""
function TreeTensorNetwork(tensors)
    itn = ITensorNetwork(tensors)
    @assert is_tree(itn)
    V = vertextype(itn)
    return TreeTensorNetwork{V}(
        itn.graph, itn.vertex_data, itn.ind_to_vertices, Indices{V}(vertices(itn))
    )
end

const TTN = TreeTensorNetwork

# Field access
"""
    ortho_region(tn::TreeTensorNetwork) -> Indices

Return the set of vertices that currently form the orthogonality center of `tn`.

See also: [`orthogonalize`](@ref).
"""
ortho_region(tn::TTN) = tn.ortho_region

# `AbstractITensorNetwork` storage hooks.
DataGraphs.underlying_graph(tn::TTN) = tn.graph
DataGraphs.vertex_data(tn::TTN) = tn.vertex_data

function DataGraphs.set_vertex_data!(tn::TTN, value, v)
    _set_vertex_data!(tn.graph, tn.vertex_data, tn.ind_to_vertices, value, v)
    return tn
end

function Graphs.rem_vertex!(tn::TTN, v)
    _rem_vertex!(tn.graph, tn.vertex_data, tn.ind_to_vertices, v)
    return tn
end

function Base.copy(tn::TTN{V}) where {V}
    return TreeTensorNetwork{V}(
        copy(tn.graph),
        map(copy, tn.vertex_data),
        Dict{Index, Set{V}}(i => copy(vs) for (i, vs) in tn.ind_to_vertices),
        copy(tn.ortho_region)
    )
end

# set_ortho_region: low-level update of the ortho_region metadata only,
# without any gauge transformations. To move the orthogonality center use orthogonalize.
function set_ortho_region(tn::TTN{V}, ortho_region) where {V}
    return TreeTensorNetwork{V}(
        tn.graph, tn.vertex_data, tn.ind_to_vertices, Indices{V}(ortho_region)
    )
end
