using .ITensorsExtensions: trivial_space
using DataGraphs: DataGraphs, DataGraph
using Dictionaries: Indices
using ITensors: ITensors, ITensor, op
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

**From an `IndsNetwork`** (most common in legacy code):

```julia
ITensorNetwork(is::IndsNetwork; link_space = 1)
ITensorNetwork(f, is::IndsNetwork; link_space = 1)
ITensorNetwork(eltype, undef, is::IndsNetwork; link_space = 1)
```

  - With no function argument `f`, tensors are initialized to zero.
  - With a function `f(v)` that returns a state label (e.g. `"Up"`, `"Dn"`) or
    an `ITensor` constructor, tensors are initialized accordingly.
  - `link_space` controls the bond-index dimension (default 1).

**From a graph (site indices inferred as trivial):**

```julia
ITensorNetwork(graph::AbstractNamedGraph; link_space = 1)
ITensorNetwork(f, graph::AbstractNamedGraph; link_space = 1)
```

# Example

```jldoctest
julia> using NamedGraphs.NamedGraphGenerators: named_grid

julia> g = named_grid((4,));

julia> s = siteinds("S=1/2", g);

julia> tn = ITensorNetwork(s; link_space = 2);

julia> tn = ITensorNetwork("Up", s);

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
function ITensorNetwork{V}(g::NamedGraph) where {V}
    return ITensorNetwork(NamedGraph{V}(g))
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

#
# Construction from underyling named graph
#

function ITensorNetwork(
        eltype::Type, undef::UndefInitializer, graph::AbstractNamedGraph; kwargs...
    )
    return ITensorNetwork(eltype, undef, IndsNetwork(graph; kwargs...))
end

function ITensorNetwork(f, graph::AbstractNamedGraph; kwargs...)
    return ITensorNetwork(f, IndsNetwork(graph; kwargs...))
end

function ITensorNetwork(graph::AbstractNamedGraph; kwargs...)
    return ITensorNetwork(IndsNetwork(graph; kwargs...))
end

#
# Construction from underyling simple graph
#

function ITensorNetwork(
        eltype::Type, undef::UndefInitializer, graph::AbstractSimpleGraph; kwargs...
    )
    return ITensorNetwork(eltype, undef, IndsNetwork(graph; kwargs...))
end

function ITensorNetwork(f, graph::AbstractSimpleGraph; kwargs...)
    return ITensorNetwork(f, IndsNetwork(graph); kwargs...)
end

function ITensorNetwork(graph::AbstractSimpleGraph; kwargs...)
    return ITensorNetwork(IndsNetwork(graph); kwargs...)
end

#
# Construction from IndsNetwork
#

function ITensorNetwork(eltype::Type, undef::UndefInitializer, is::IndsNetwork; kwargs...)
    return ITensorNetwork(is; kwargs...) do v
        return (inds...) -> ITensor(eltype, undef, inds...)
    end
end

function ITensorNetwork(eltype::Type, is::IndsNetwork; kwargs...)
    return ITensorNetwork(is; kwargs...) do v
        return (inds...) -> ITensor(eltype, inds...)
    end
end

function ITensorNetwork(undef::UndefInitializer, is::IndsNetwork; kwargs...)
    return ITensorNetwork(is; kwargs...) do v
        return (inds...) -> ITensor(undef, inds...)
    end
end

function ITensorNetwork(is::IndsNetwork; kwargs...)
    return ITensorNetwork(is; kwargs...) do v
        return (inds...) -> ITensor(inds...)
    end
end

# TODO: Handle `eltype` and `undef` through `generic_state`.
# `inds` are stored in a `NamedTuple`
function generic_state(f, inds::NamedTuple)
    return generic_state(f, reduce(vcat, inds.linkinds; init = inds.siteinds))
end

function generic_state(f, inds::Vector)
    return f(inds)
end
function generic_state(a::AbstractArray, inds::Vector)
    return itensor(a, inds)
end
function generic_state(x::Op, inds::NamedTuple)
    # TODO: Figure out what to do if there is more than one site.
    if !isempty(inds.siteinds)
        @assert length(inds.siteinds) == 2
        i = inds.siteinds[findfirst(i -> plev(i) == 0, inds.siteinds)]
        @assert i' ∈ inds.siteinds
        site_tensors = [op(x.which_op, i)]
    else
        site_tensors = []
    end
    link_tensors = [[onehot(i => 1) for i in inds.linkinds[e]] for e in keys(inds.linkinds)]
    return contract(reduce(vcat, link_tensors; init = site_tensors))
end
function generic_state(s::AbstractString, inds::NamedTuple)
    # TODO: Figure out what to do if there is more than one site.
    site_tensors = [ITensors.state(s, only(inds.siteinds))]
    link_tensors = [[onehot(i => 1) for i in inds.linkinds[e]] for e in keys(inds.linkinds)]
    return contract(reduce(vcat, link_tensors; init = site_tensors))
end

# TODO: This is similar to `ModelHamiltonians.to_callable`,
# try merging the two.
to_callable(value::Type) = value
to_callable(value::Function) = value
function to_callable(value::AbstractDict)
    return Base.Fix1(getindex, value) ∘ keytype(value)
end
function to_callable(value::AbstractDictionary)
    return Base.Fix1(getindex, value) ∘ keytype(value)
end
function to_callable(value::AbstractArray{<:Any, N}) where {N}
    return Base.Fix1(getindex, value) ∘ CartesianIndex
end
to_callable(value) = Returns(value)

function ITensorNetwork(value, is::IndsNetwork; kwargs...)
    return ITensorNetwork(to_callable(value), is; kwargs...)
end

function ITensorNetwork(
        elt::Type, f, is::IndsNetwork; link_space = trivial_space(is), kwargs...
    )
    tn = ITensorNetwork(f, is; kwargs...)
    for v in vertices(tn)
        # TODO: Ideally we would use broadcasting, i.e. `elt.(tn[v])`,
        # but that doesn't work right now on ITensors.
        tn[v] = ITensors.convert_eltype(elt, tn[v])
    end
    return tn
end

function ITensorNetwork(
        itensor_constructor::Function, is::IndsNetwork; link_space = trivial_space(is),
        kwargs...
    )
    is = insert_linkinds(is; link_space)
    tn = ITensorNetwork{vertextype(is)}(ITensor[])
    for v in vertices(is)
        add_vertex!(tn, v)
    end
    for e in edges(is)
        add_edge!(tn, e)
    end
    for v in vertices(tn)
        # TODO: Replace with `is[v]` once `getindex(::IndsNetwork, ...)` is smarter.
        siteinds = get(is, v, Index[])
        edges = [edgetype(is)(v, nv) for nv in neighbors(is, v)]
        linkinds = map(e -> is[e], Indices(edges))
        tensor_v = generic_state(itensor_constructor(v), (; siteinds, linkinds))
        setindex_preserve_graph!(tn, tensor_v, v)
    end
    return tn
end
