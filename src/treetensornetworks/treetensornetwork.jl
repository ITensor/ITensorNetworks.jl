using Dictionaries: Indices
using Graphs: path_graph
using ITensors: ITensor
using LinearAlgebra: factorize, normalize
using NamedGraphs.GraphsExtensions: GraphsExtensions, vertextype

"""
    TreeTensorNetwork{V} <: AbstractTreeTensorNetwork{V}

A tensor network whose underlying graph is a tree. In addition to the tensor data,
it tracks an `ortho_region`: the set of vertices that currently form the orthogonality
center of the network.

`TTN` is an alias for `TreeTensorNetwork`.

Use [`ttn`](@ref) or [`mps`](@ref) to construct instances, and [`orthogonalize`](@ref) to
bring the network into a canonical gauge.

See also: [`ITensorNetwork`](@ref), [`ttn`](@ref), [`mps`](@ref), [`random_ttn`](@ref).
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
```julia
ttn_state = TreeTensorNetwork(itn; ortho_region = [root_vertex])
```

See also: [`ttn`](@ref), [`ITensorNetwork`](@ref), [`orthogonalize`](@ref).
"""
function TreeTensorNetwork(tn::ITensorNetwork; ortho_region = vertices(tn))
    return _TreeTensorNetwork(tn, ortho_region)
end
function TreeTensorNetwork{V}(tn::ITensorNetwork) where {V}
    return TreeTensorNetwork(ITensorNetwork{V}(tn))
end

const TTN = TreeTensorNetwork

# Field access
"""
    ITensorNetwork(tn::TreeTensorNetwork) -> ITensorNetwork

Convert a `TreeTensorNetwork` to a plain `ITensorNetwork`, discarding orthogonality
metadata. The returned network shares the same underlying tensor data.

See also: [`TreeTensorNetwork`](@ref), [`ttn`](@ref).
"""
ITensorNetwork(tn::TTN) = getfield(tn, :tensornetwork)

"""
    ortho_region(tn::TreeTensorNetwork) -> Indices

Return the set of vertices that currently form the orthogonality center of `tn`.

See also: [`orthogonalize`](@ref), [`set_ortho_region`](@ref).
"""
ortho_region(tn::TTN) = getfield(tn, :ortho_region)

# Required for `AbstractITensorNetwork` interface
data_graph(tn::TTN) = data_graph(ITensorNetwork(tn))

function data_graph_type(G::Type{<:TTN})
    return data_graph_type(fieldtype(G, :tensornetwork))
end

function Base.copy(tn::TTN)
    return _TreeTensorNetwork(copy(ITensorNetwork(tn)), copy(ortho_region(tn)))
end

#
# Constructor
#

"""
    set_ortho_region(tn::TreeTensorNetwork, ortho_region) -> TreeTensorNetwork

Return a copy of `tn` with the `ortho_region` metadata updated to `ortho_region`,
**without** performing any gauge transformations on the tensors.

This is a low-level bookkeeping update. To actually move the orthogonality center by
applying QR decompositions along the tree, use [`orthogonalize`](@ref).

See also: [`ortho_region`](@ref), [`orthogonalize`](@ref).
"""
function set_ortho_region(tn::TTN, ortho_region)
    return ttn(ITensorNetwork(tn); ortho_region)
end

"""
    ttn(args...; ortho_region=nothing) -> TreeTensorNetwork

Construct a `TreeTensorNetwork` (TTN) using the same interface as [`ITensorNetwork`](@ref).
All positional and keyword arguments are forwarded to the `ITensorNetwork` constructor.

If `ortho_region` is not specified, all vertices are set as the orthogonal region
(i.e. no particular gauge is assumed). Call [`orthogonalize`](@ref) to impose a gauge.

# Example
```julia
using ITensorNetworks, NamedGraphs.NamedGraphGenerators

# Comb-tree TTN with random bond-dimension-2 tensors
g = named_comb_tree((3, 4))
s = siteinds("S=1/2", g)
psi = ttn(v -> "Up", s; link_space = 2)
```

See also: [`mps`](@ref), [`random_ttn`](@ref), [`TreeTensorNetwork`](@ref).
"""
function ttn(args...; ortho_region = nothing)
    tn = ITensorNetwork(args...)
    if isnothing(ortho_region)
        ortho_region = vertices(tn)
    end
    return _TreeTensorNetwork(tn, ortho_region)
end

"""
    mps(args...; ortho_region=nothing) -> TreeTensorNetwork

Construct a matrix product state (MPS) as a `TreeTensorNetwork` on a 1D path graph.
The interface is identical to [`ttn`](@ref) but is intended for 1D (chain) topologies.

See also: [`ttn`](@ref), [`random_mps`](@ref).
"""
function mps(args...; ortho_region = nothing)
    # TODO: Check it is a path graph.
    tn = ITensorNetwork(args...)
    if isnothing(ortho_region)
        ortho_region = vertices(tn)
    end
    return _TreeTensorNetwork(tn, ortho_region)
end

"""
    mps(f, is::Vector{<:Index}; kwargs...) -> TreeTensorNetwork

Construct a matrix product state (MPS) from a function `f` and a flat vector of site
indices `is`. The indices are arranged on a 1D path graph automatically.

# Example
```julia
s = siteinds("S=1/2", 10)
psi = mps(v -> "Up", s)
```
"""
function mps(f, is::Vector{<:Index}; kwargs...)
    return mps(f, path_indsnetwork(is); kwargs...)
end

"""
    ttn(a::ITensor, is::IndsNetwork; ortho_region=..., kwargs...) -> TreeTensorNetwork

Decompose a dense `ITensor` `a` into a `TreeTensorNetwork` with the tree structure
described by the `IndsNetwork` `is`.

Successive QR/SVD factorizations are applied following a post-order DFS traversal from the
root vertex, then the network is orthogonalized to `ortho_region` (defaults to the root).
Extra `kwargs` (e.g. `cutoff`, `maxdim`) are forwarded to the factorization.

# Example
```julia
i, j, k = Index(2, "i"), Index(2, "j"), Index(2, "k")
A = randomITensor(i, j, k)
is = IndsNetwork(named_comb_tree((3,)); site_space = [i, j, k])
ttn_A = ttn(A, is)
```
"""
# Construct from dense ITensor, using IndsNetwork of site indices.
function ttn(
        a::ITensor,
        is::IndsNetwork;
        ortho_region = Indices([GraphsExtensions.default_root_vertex(is)]),
        kwargs...
    )
    for v in vertices(is)
        @assert hasinds(a, is[v])
    end
    @assert ortho_region ⊆ vertices(is)
    tn = ITensorNetwork(is)
    ortho_center = first(ortho_region)
    for e in post_order_dfs_edges(tn, ortho_center)
        left_inds = uniqueinds(is, e)
        a_l, a_r = factorize(a, left_inds; tags = edge_tag(e), ortho = "left", kwargs...)
        tn[src(e)] = a_l
        is[e] = commoninds(a_l, a_r)
        a = a_r
    end
    tn[ortho_center] = a
    ttn_a = ttn(tn)
    return orthogonalize(ttn_a, ortho_center)
end

"""
    random_ttn(args...; kwargs...) -> TreeTensorNetwork

Construct a random, unit-norm `TreeTensorNetwork`. Arguments are forwarded to
`random_tensornetwork`, which accepts the same interface as [`ITensorNetwork`](@ref).

# Example
```julia
g = named_comb_tree((3, 4))
s = siteinds("S=1/2", g)
psi = random_ttn(s; link_space = 4)
```

See also: [`ttn`](@ref), [`random_mps`](@ref).
"""
function random_ttn(args...; kwargs...)
    # TODO: Check it is a tree graph.
    return normalize(_TreeTensorNetwork(random_tensornetwork(args...; kwargs...)))
end

"""
    random_mps(args...; kwargs...) -> TreeTensorNetwork

Construct a random, unit-norm matrix product state (MPS) as a `TreeTensorNetwork`.
Arguments are forwarded to [`random_ttn`](@ref).

# Example
```julia
s = siteinds("S=1/2", 10)
psi = random_mps(s; link_space = 4)
```

See also: [`mps`](@ref), [`random_ttn`](@ref).
"""
function random_mps(args...; kwargs...)
    # TODO: Check it is a path graph.
    return random_ttn(args...; kwargs...)
end

"""
    random_mps(f, is::Vector{<:Index}; kwargs...) -> TreeTensorNetwork

Construct a random MPS from a function `f` and a flat vector of site indices `is`.
"""
function random_mps(f, is::Vector{<:Index}; kwargs...)
    return random_mps(f, path_indsnetwork(is); kwargs...)
end

"""
    random_mps(s::Vector{<:Index}; kwargs...) -> TreeTensorNetwork

Construct a random MPS from a flat vector of site indices `s`.
"""
function random_mps(s::Vector{<:Index}; kwargs...)
    return random_mps(path_indsnetwork(s); kwargs...)
end
