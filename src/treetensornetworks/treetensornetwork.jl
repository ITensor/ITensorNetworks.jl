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

"""
    TreeTensorNetwork(tn::ITensorNetwork; ortho_region=vertices(tn)) -> TreeTensorNetwork

Construct a `TreeTensorNetwork` from an `ITensorNetwork` with tree graph structure.

The `ortho_region` keyword specifies which vertices currently form the orthogonality center.
By default all vertices are included, meaning no particular gauge is assumed. To enforce an
actual orthogonal gauge, call [`orthogonalize`](@ref) afterward.

Throws an error if the underlying graph of `tn` is not a tree.

# Example

```jldoctest
julia> using ITensors: Index, ITensor

julia> using Graphs: vertices

julia> i, j, k = Index(2, "i"), Index(2, "j"), Index(2, "k");

julia> itn = ITensorNetwork([ITensor(i, j), ITensor(j, k)]);

julia> ttn = TreeTensorNetwork(itn; ortho_region = [first(vertices(itn))]);

```

See also: [`ITensorNetwork`](@ref), [`orthogonalize`](@ref).
"""
function TreeTensorNetwork(tn::ITensorNetwork{V}; ortho_region = vertices(tn)) where {V}
    @assert is_tree(tn)
    return TreeTensorNetwork{V}(
        tn.graph, tn.vertex_data, tn.ind_to_vertices, Indices{V}(ortho_region)
    )
end

function TreeTensorNetwork{V}(tn::ITensorNetwork) where {V}
    return TreeTensorNetwork(ITensorNetwork{V}(tn))
end

# Build a `TreeTensorNetwork` directly from a tensor collection (anything
# accepted by `ITensorNetwork`), saving the caller a wrapping step.
function TreeTensorNetwork(tensors; kwargs...)
    return TreeTensorNetwork(ITensorNetwork(tensors); kwargs...)
end

const TTN = TreeTensorNetwork

# Field access
"""
    ITensorNetwork(tn::TreeTensorNetwork) -> ITensorNetwork

Convert a `TreeTensorNetwork` to a plain `ITensorNetwork`, discarding orthogonality
metadata. The returned network shares the same underlying tensor data.

See also: [`TreeTensorNetwork`](@ref).
"""
ITensorNetwork(tn::TTN) = ITensorNetwork(map(copy, vertex_data(tn)))

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
