using Dictionaries: Indices
using NamedGraphs.GraphsExtensions: vertextype
using NamedGraphs: similar_graph

"""
    TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}

A tensor network whose underlying graph is a tree. In addition to the tensor data,
it tracks an `ortho_region`: the set of vertices that currently form the orthogonality
center of the network.

`TTN` is an alias for `TreeTensorNetwork`.

Use the [`TreeTensorNetwork`](@ref) constructors to build instances, and
[`orthogonalize`](@ref) to bring the network into a canonical gauge.

See also: [`ITensorNetwork`](@ref).
"""
struct TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}
    tensornetwork::ITensorNetwork{V}
    ortho_region::Indices{V}
    global function _TreeTensorNetwork(tensornetwork::ITensorNetwork, ortho_region::Indices)
        @assert is_tree(tensornetwork)
        return new{vertextype(tensornetwork)}(tensornetwork, ortho_region)
    end
end

function _TreeTensorNetwork(tensornetwork::ITensorNetwork, ortho_region)
    return _TreeTensorNetwork(tensornetwork, Indices(ortho_region))
end

function _TreeTensorNetwork(tensornetwork::ITensorNetwork)
    return _TreeTensorNetwork(tensornetwork, vertices(tensornetwork))
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
julia> using NamedGraphs.NamedGraphGenerators: named_comb_tree

julia> using NamedGraphs: NamedGraph

julia> using Graphs: vertices

julia> using ITensors: ITensor

julia> g = named_comb_tree((2, 2));

julia> s = siteinds("S=1/2", g);

julia> tensors = Dict(v => ITensor(s[v]...) for v in vertices(g));

julia> itn = ITensorNetwork(tensors, NamedGraph(g));

julia> ttn_state = TreeTensorNetwork(itn; ortho_region = [first(vertices(itn))]);

```

See also: [`ITensorNetwork`](@ref), [`orthogonalize`](@ref).
"""
function TreeTensorNetwork(tn::ITensorNetwork; ortho_region = vertices(tn))
    return _TreeTensorNetwork(tn, ortho_region)
end
function TreeTensorNetwork{V}(tn::ITensorNetwork) where {V}
    return TreeTensorNetwork(ITensorNetwork{V}(tn))
end

const TTN = TreeTensorNetwork

function NamedGraphs.similar_graph(ttn::TTN, underlying_graph::AbstractGraph)
    return TTN(similar_graph(ttn.tensornetwork, underlying_graph))
end

# Field access
"""
    ITensorNetwork(tn::TreeTensorNetwork) -> ITensorNetwork

Convert a `TreeTensorNetwork` to a plain `ITensorNetwork`, discarding orthogonality
metadata. The returned network shares the same underlying tensor data.

See also: [`TreeTensorNetwork`](@ref).
"""
ITensorNetwork(tn::TTN) = copy(tn.tensornetwork)

"""
    ortho_region(tn::TreeTensorNetwork) -> Indices

Return the set of vertices that currently form the orthogonality center of `tn`.

See also: [`orthogonalize`](@ref).
"""
ortho_region(tn::TTN) = tn.ortho_region

# Required for `AbstractITensorNetwork` interface
data_graph(tn::TTN) = data_graph(tn.tensornetwork)

function data_graph_type(G::Type{<:TTN})
    return data_graph_type(fieldtype(G, :tensornetwork))
end

function Base.copy(tn::TTN)
    return _TreeTensorNetwork(copy(tn.tensornetwork), copy(tn.ortho_region))
end

#
# Constructor
#

# set_ortho_region: low-level update of the ortho_region metadata only,
# without any gauge transformations. To move the orthogonality center use orthogonalize.
function set_ortho_region(tn::TTN, ortho_region)
    return TreeTensorNetwork(tn.tensornetwork; ortho_region)
end
