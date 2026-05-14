using Adapt: Adapt, adapt, adapt_structure
using DataGraphs:
    DataGraphs, set_vertex_data!, underlying_graph, underlying_graph_type, vertex_data
using Dictionaries: Dictionaries, Dictionary
using Graphs: Graphs, Graph, add_edge!, add_vertex!, bfs_tree, center, dst, edges, edgetype,
    has_edge, ne, neighbors, rem_edge!, src, vertices
using ITensors: ITensors, ITensor, Index, addtags, commoninds, commontags, contract, dag,
    inds, noprime, onehot, prime, replaceprime, replacetags, setprime, settags, sim,
    swaptags, tags
using LinearAlgebra: LinearAlgebra, qr, qr!
using NDTensors: NDTensors, Algorithm, dim, scalartype
using NamedGraphs.GraphsExtensions:
    add_edges, directed_graph, incident_edges, rename_vertices, vertextype, ⊔
using NamedGraphs: NamedGraphs, NamedGraph, Vertices, not_implemented, steiner_tree
using SplitApplyCombine: flatten

abstract type AbstractITensorNetwork{V} <: AbstractDataGraph{V, ITensor, ITensor} end

# Subtypes provide the storage: `underlying_graph(tn)` returns the named graph
# and `vertex_data(tn)` returns a `Dictionary{V, ITensor}`-like mapping. Edge
# data is unused — every `AbstractITensorNetwork` is treated as having no edge
# data.

# TODO: Define a generic fallback for `AbstractDataGraph`?
DataGraphs.edge_data_type(::Type{<:AbstractITensorNetwork}) = ITensor

# Graphs.jl overloads
function Graphs.weights(graph::AbstractITensorNetwork)
    V = vertextype(graph)
    es = Tuple.(edges(graph))
    ws = Dictionary{Tuple{V, V}, Float64}(es, undef)
    for e in edges(graph)
        w = log2(dim(linkinds(graph, e)))
        ws[(src(e), dst(e))] = w
    end
    return ws
end

# Copy
Base.copy(tn::AbstractITensorNetwork) = not_implemented()

# Vertex-keyed access: `keys(tn)` returns the vertex set, `values(tn)` the
# tensors, and `tn[v]` the tensor at vertex `v`. Going through `values(tn)`
# (rather than `values(vertex_data(tn))`) lets callers stay agnostic about
# whether `vertex_data` is a `Dict`, `Dictionary`, or anything else with
# different default-iteration semantics.
Base.keys(tn::AbstractITensorNetwork) = vertices(tn)
Base.keytype(::Type{<:AbstractITensorNetwork{V}}) where {V} = V
Base.keytype(tn::AbstractITensorNetwork) = keytype(typeof(tn))
Base.values(tn::AbstractITensorNetwork) = (tn[v] for v in vertices(tn))
Base.iterate(tn::AbstractITensorNetwork, args...) = iterate(values(tn), args...)

# TODO: This contrasts with the `DataGraphs.AbstractDataGraph` definition,
# where it is defined as the `vertextype`. Does that cause problems or should it be changed?
Base.eltype(tn::AbstractITensorNetwork) = eltype(vertex_data(tn))

# Overload if needed
Graphs.is_directed(::Type{<:AbstractITensorNetwork}) = false
function GraphsExtensions.directed_graph(tn::AbstractITensorNetwork)
    return directed_graph(underlying_graph(tn))
end

function ITensors.datatype(tn::AbstractITensorNetwork)
    return mapreduce(v -> datatype(tn[v]), promote_type, vertices(tn))
end

# AbstractDataGraphs overloads — defined directly in terms of the
# `underlying_graph` / `vertex_data` storage, with no edge data.

function DataGraphs.is_vertex_assigned(tn::AbstractITensorNetwork, v)
    return haskey(vertex_data(tn), v)
end

DataGraphs.is_edge_assigned(::AbstractITensorNetwork, _) = false

DataGraphs.get_vertex_data(tn::AbstractITensorNetwork, v) = vertex_data(tn)[v]

function NamedGraphs.vertex_positions(tn::AbstractITensorNetwork)
    return NamedGraphs.vertex_positions(underlying_graph(tn))
end
function NamedGraphs.ordered_vertices(tn::AbstractITensorNetwork)
    return NamedGraphs.ordered_vertices(underlying_graph(tn))
end

function Adapt.adapt_structure(to, tn::AbstractITensorNetwork)
    return map(adapt(to), tn)
end

#
# Iteration
#

# TODO: iteration

# TODO: different `map` functionalities as defined for ITensors.AbstractMPS

# TODO: broadcasting

function Base.union(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kwargs...)
    g = union(underlying_graph(tn1), underlying_graph(tn2); kwargs...)
    tensors = Dict{vertextype(g), ITensor}(
        v => (v in vertices(tn1) ? tn1[v] : tn2[v]) for v in vertices(g)
    )
    # `ITensorNetwork(tensors)` infers edges from shared `Index`es, so any
    # cross-network bonds between `tn1` and `tn2` are picked up automatically.
    return ITensorNetwork(tensors)
end

function NamedGraphs.rename_vertices(f::Function, tn::AbstractITensorNetwork)
    return ITensorNetwork(Dict(f(v) => tn[v] for v in vertices(tn)))
end

#
# Data modification
#

#
# Promotion and conversion
#

function NDTensors.scalartype(tn::AbstractITensorNetwork)
    return mapreduce(v -> eltype(tn[v]), promote_type, vertices(tn); init = Bool)
end

# TODO: Define `eltype(::AbstractITensorNetwork)` as `ITensor`?

function NDTensors.convert_scalartype(eltype::Type{<:Number}, tn::AbstractITensorNetwork)
    return map(adapt(eltype), tn)
end

function Base.complex(tn::AbstractITensorNetwork)
    return NDTensors.convert_scalartype(complex(scalartype(tn)), tn)
end

#
# Conversion to Graphs
#

function Graphs.Graph(tn::AbstractITensorNetwork)
    return Graph(Vector{ITensor}(tn))
end

function NamedGraphs.NamedGraph(tn::AbstractITensorNetwork)
    return NamedGraph(Vector{ITensor}(tn))
end

#
# Conversion to IndsNetwork
#

# Convert to an IndsNetwork
function IndsNetwork(tn::AbstractITensorNetwork)
    is = IndsNetwork(underlying_graph(tn))
    for v in vertices(tn)
        is[v] = siteinds(tn, v)
    end
    for e in edges(tn)
        is[e] = linkinds(tn, e)
    end
    return is
end

# TODO: Output a `VertexDataGraph`? Unfortunately
# `IndsNetwork` doesn't allow iterating over vertex data.
function siteinds(tn::AbstractITensorNetwork)
    is = IndsNetwork(underlying_graph(tn))
    for v in vertices(tn)
        is[v] = siteinds(tn, v)
    end
    return is
end

function linkinds(tn::AbstractITensorNetwork)
    is = IndsNetwork(underlying_graph(tn))
    for e in edges(tn)
        is[e] = linkinds(tn, e)
    end
    return is
end

#
# Index access
#

function _siteinds(tn::AbstractITensorNetwork, vertex)
    s = collect(inds(tn[vertex]))
    for v in neighbors(tn, vertex)
        s = setdiff(s, inds(tn[v]))
    end
    return s
end
siteinds(tn::AbstractITensorNetwork, vertex) = _siteinds(tn, vertex)
# Fix ambiguity with `siteinds(::Type, ::Int)` from `sitetype.jl`.
siteinds(tn::AbstractITensorNetwork, vertex::Int) = _siteinds(tn, vertex)

function linkinds(tn::AbstractITensorNetwork, edge)
    e = edgetype(tn)(edge)
    return intersect(inds(tn[src(e)]), inds(tn[dst(e)]))
end

# Priming and tagging (changing Index identifiers)
function ITensors.replaceinds(
        tn::AbstractITensorNetwork, is_is′::Pair{<:IndsNetwork, <:IndsNetwork}
    )
    tn = copy(tn)
    is, is′ = is_is′
    @assert underlying_graph(is) == underlying_graph(is′)
    for v in vertices(is)
        isassigned(is, v) || continue
        tn[v] = replaceinds(tn[v], is[v] => is′[v])
    end
    for e in edges(is)
        isassigned(is, e) || continue
        for v in (src(e), dst(e))
            tn[v] = replaceinds(tn[v], is[e] => is′[e])
        end
    end
    return tn
end

function map_inds(f, tn::AbstractITensorNetwork, args...; kwargs...)
    is = IndsNetwork(tn)
    is′ = map_inds(f, is, args...; kwargs...)
    return replaceinds(tn, is => is′)
end

const map_inds_label_functions = [
    :prime,
    :setprime,
    :noprime,
    :replaceprime,
    # :swapprime, # TODO: add @test_broken as a reminder
    :addtags,
    :removetags,
    :replacetags,
    :settags,
    :sim,
    :swaptags,
    :dag,
    # :replaceind,
    # :replaceinds,
    # :swapind,
    # :swapinds,
]

for f in map_inds_label_functions
    @eval begin
        function ITensors.$f(
                n::Union{IndsNetwork, AbstractITensorNetwork},
                args...;
                kwargs...
            )
            return map_inds($f, n, args...; kwargs...)
        end

        function ITensors.$f(
                ffilter::typeof(linkinds),
                n::Union{IndsNetwork, AbstractITensorNetwork},
                args...;
                kwargs...
            )
            return map_inds($f, n, args...; sites = [], kwargs...)
        end

        function ITensors.$f(
                ffilter::typeof(siteinds),
                n::Union{IndsNetwork, AbstractITensorNetwork},
                args...;
                kwargs...
            )
            return map_inds($f, n, args...; links = [], kwargs...)
        end
    end
end

LinearAlgebra.adjoint(tn::Union{IndsNetwork, AbstractITensorNetwork}) = prime(tn)

# In-place / out-of-place `map` over the vertex tensors of `tn`. Reverse-map
# reconciliation makes the per-vertex write cheap, so these match
# `Base.map` / `Base.map!`'s usual element-wise semantics without needing
# a separate "preserve graph" entry point.
function Base.map!(f, tn::AbstractITensorNetwork)
    for v in vertices(tn)
        tn[v] = f(tn[v])
    end
    return tn
end

Base.map(f, tn::AbstractITensorNetwork) = map!(f, copy(tn))

function Base.conj(tn::AbstractITensorNetwork)
    return map(conj, tn)
end

function ITensors.dag(tn::AbstractITensorNetwork)
    return map(dag, tn)
end

# TODO: should this make sure that internal indices
# don't clash?
function ⊗(
        tn1::AbstractITensorNetwork,
        tn2::AbstractITensorNetwork,
        tn_tail::AbstractITensorNetwork...;
        kwargs...
    )
    return ⊔(tn1, tn2, tn_tail...; kwargs...)
end

function ⊗(
        tn1::Pair{<:Any, <:AbstractITensorNetwork},
        tn2::Pair{<:Any, <:AbstractITensorNetwork},
        tn_tail::Pair{<:Any, <:AbstractITensorNetwork}...;
        kwargs...
    )
    return ⊔(tn1, tn2, tn_tail...; kwargs...)
end

# TODO: how to define this lazily?
#norm(tn::AbstractITensorNetwork) = sqrt(inner(tn, tn))

function Base.isapprox(
        x::AbstractITensorNetwork,
        y::AbstractITensorNetwork;
        atol::Real = 0,
        rtol::Real = Base.rtoldefault(scalartype(x), scalartype(y), atol)
    )
    error("Not implemented")
    d = norm(x - y)
    if !isfinite(d)
        error(
            "In `isapprox(x::AbstractITensorNetwork, y::AbstractITensorNetwork)`, `norm(x - y)` is not finite"
        )
    end
    return d <= max(atol, rtol * max(norm(x), norm(y)))
end

function ITensors.contract(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
    return contract(tn, edgetype(tn)(edge); kwargs...)
end

# Contract the tensors at vertices `src(edge)` and `dst(edge)`
# and store the results in the vertex `dst(edge)`, removing
# the vertex `src(edge)`.
# TODO: write this in terms of a more generic function
# `Graphs.merge_vertices!` (https://github.com/mtfishman/ITensorNetworks.jl/issues/12)
function NDTensors.contract(
        tn::AbstractITensorNetwork, edge::AbstractEdge; merged_vertex = dst(edge)
    )
    V = promote_type(vertextype(tn), typeof(merged_vertex))
    # TODO: Check `ITensorNetwork{V}`, shouldn't need a copy here.
    tn = ITensorNetwork{V}(copy(tn))
    new_itensor = tn[src(edge)] * tn[dst(edge)]
    rem_vertex!(tn, src(edge))
    rem_vertex!(tn, dst(edge))
    # `setindex!` (via `set_vertex_data!`) adds `merged_vertex` to the
    # graph and reverse-map reconciliation picks up the new bonds to the
    # surviving neighbors of `src(edge)` and `dst(edge)`.
    tn[merged_vertex] = new_itensor
    return tn
end

function ITensors.tags(tn::AbstractITensorNetwork, edge)
    is = linkinds(tn, edge)
    return commontags(is)
end

# QR-factor `tn[src(edge)]` along the inds it doesn't share with
# `tn[dst(edge)]`, keep the orthogonal factor `Q` on `src(edge)`, and
# absorb the residual `R` into `tn[dst(edge)]`. Mutates `tn` in place;
# the graph is unchanged.
function LinearAlgebra.qr!(tn::AbstractITensorNetwork, edge::AbstractEdge)
    left_inds = setdiff(inds(tn[src(edge)]), inds(tn[dst(edge)]))
    Q, R = qr(tn[src(edge)], left_inds; tags = edge_tag(edge))
    tn[src(edge)] = Q
    tn[dst(edge)] = R * tn[dst(edge)]
    return tn
end

function LinearAlgebra.qr!(tn::AbstractITensorNetwork, edge::Pair)
    return qr!(tn, edgetype(tn)(edge))
end

LinearAlgebra.qr(tn::AbstractITensorNetwork, edge::AbstractEdge) = qr!(copy(tn), edge)
LinearAlgebra.qr(tn::AbstractITensorNetwork, edge::Pair) = qr!(copy(tn), edge)

function gauge_walk(tn::AbstractITensorNetwork, edges)
    tn = copy(tn)
    for edge in edges
        qr!(tn, edge)
    end
    return tn
end

tree_gauge(ψ::AbstractITensorNetwork, region) = tree_gauge(ψ, [region])

#Get the path that moves the gauge from a to b on a tree
#TODO: Move to NamedGraphs
function edge_sequence_between_regions(g::AbstractGraph, region_a::Vector, region_b::Vector)
    issetequal(region_a, region_b) && return edgetype(g)[]
    st = steiner_tree(g, union(region_a, region_b))
    path = post_order_dfs_edges(st, first(region_b))
    path = filter(e -> !((src(e) ∈ region_b) && (dst(e) ∈ region_b)), path)
    return path
end

# Gauge a ITensorNetwork from cur_region towards new_region, treating
# the network as a tree spanned by a spanning tree.
function tree_gauge(ψ::AbstractITensorNetwork, cur_region::Vector, new_region::Vector)
    es = edge_sequence_between_regions(ψ, cur_region, new_region)
    return gauge_walk(ψ, es)
end

# Gauge a ITensorNetwork towards a region, treating
# the network as a tree spanned by a spanning tree.
function tree_gauge(ψ::AbstractITensorNetwork, region::Vector)
    return tree_gauge(ψ, collect(vertices(ψ)), region)
end

tree_orthogonalize(ψ::AbstractITensorNetwork, args...) = tree_gauge(ψ, args...)

"""
    left_orth!(tn::AbstractITensorNetwork, edge; kwargs...) -> AbstractITensorNetwork

Mutate `tn` so the bond at `edge` is left-orthogonal: SVD-factorize
`tn[src(edge)]` into `U * S * V`, replace `tn[src(edge)]` with the
left-isometric `U`, and absorb `S * V` into `tn[dst(edge)]`. `edge` may be an
`AbstractEdge` or a `Pair` of vertices.

`kwargs...` are forwarded to `ITensors.svd`, so passing `cutoff` / `maxdim` /
`mindim` truncates the bond in addition to gauging it. For a non-truncating
left-orthogonal gauge, prefer `LinearAlgebra.qr!(tn, edge)`, which uses QR
instead of SVD.

See also: [`left_orth`](@ref), [`Base.truncate(::AbstractITensorNetwork, ::AbstractEdge)`](@ref).
"""
function left_orth!(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
    !has_edge(tn, edge) && throw(ArgumentError("Edge not in graph."))
    left_inds = setdiff(inds(tn[src(edge)]), inds(tn[dst(edge)]))
    ltags = tags(tn, edge)
    U, S, V = svd(tn[src(edge)], left_inds; lefttags = ltags, kwargs...)
    tn[src(edge)] = U
    tn[dst(edge)] = tn[dst(edge)] * (S * V)
    return tn
end

function left_orth!(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
    return left_orth!(tn, edgetype(tn)(edge); kwargs...)
end

"""
    left_orth(tn::AbstractITensorNetwork, edge; kwargs...) -> AbstractITensorNetwork

Non-mutating version of [`left_orth!`](@ref): return a copy of `tn` with the
bond at `edge` made left-orthogonal.
"""
function left_orth(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
    return left_orth!(copy(tn), edge; kwargs...)
end

function left_orth(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
    return left_orth(tn, edgetype(tn)(edge); kwargs...)
end

"""
    truncate(tn::AbstractITensorNetwork, edge; kwargs...) -> AbstractITensorNetwork

Truncate the bond across `edge` in `tn` by performing an SVD and discarding small
singular values. `edge` may be an `AbstractEdge` or a `Pair` of vertices.

This is the user-facing verb for bond compression and delegates to
[`left_orth`](@ref), which names the underlying left-orthogonal gauge
transformation. Truncation parameters are passed as keyword arguments and
forwarded to `ITensors.svd`:

  - `cutoff`: Drop singular values smaller than this threshold.
  - `maxdim`: Maximum number of singular values to keep.
  - `mindim`: Minimum number of singular values to keep.

This operates on a single bond. For `TreeTensorNetwork`, the no-argument form
`truncate(tn; kwargs...)` sweeps all bonds and is generally preferred for full
recompression after addition or subspace expansion.

See also: [`left_orth`](@ref), `Base.truncate(::AbstractTreeTensorNetwork)`.
"""
function Base.truncate(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
    return left_orth(tn, edge; kwargs...)
end

function Base.truncate(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
    return truncate(tn, edgetype(tn)(edge); kwargs...)
end

function Base.:*(c::Number, ψ::AbstractITensorNetwork)
    v₁ = first(vertices(ψ))
    cψ = copy(ψ)
    cψ[v₁] *= c
    return cψ
end

# Return a list of vertices in the ITensorNetwork `ψ`
# that share indices with the ITensor `T`
function neighbor_vertices(ψ::AbstractITensorNetwork, T::ITensor)
    ψT = ψ ⊔ ITensorNetwork([T])
    v⃗ = neighbors(ψT, (1, 2))
    return first.(v⃗)
end

function inner_network(x::AbstractITensorNetwork, y::AbstractITensorNetwork; kwargs...)
    return LinearFormNetwork(x, y; kwargs...)
end

function inner_network(
        x::AbstractITensorNetwork, A::AbstractITensorNetwork, y::AbstractITensorNetwork;
        kwargs...
    )
    return BilinearFormNetwork(A, x, y; kwargs...)
end

norm_sqr_network(ψ::AbstractITensorNetwork) = QuadraticFormNetwork(ψ)

#
# Printing
#

function Base.show(io::IO, mime::MIME"text/plain", graph::AbstractITensorNetwork)
    println(io, "$(typeof(graph)) with $(nv(graph)) vertices:")
    show(io, mime, vertices(graph))
    println(io, "\n")
    println(io, "and $(ne(graph)) edge(s):")
    for e in edges(graph)
        show(io, mime, e)
        println(io)
    end
    println(io)
    println(io, "with vertex data:")
    show(io, mime, inds.(vertex_data(graph)))
    return nothing
end

Base.show(io::IO, graph::AbstractITensorNetwork) = show(io, MIME"text/plain"(), graph)

#
# Link dimensions
#

function maxlinkdim(tn::AbstractITensorNetwork)
    md = 1
    for e in edges(tn)
        md = max(md, linkdim(tn, e))
    end
    return md
end

function linkdim(tn::AbstractITensorNetwork, edge::Pair)
    return linkdim(tn, edgetype(tn)(edge))
end

function linkdim(tn::AbstractITensorNetwork{V}, edge::AbstractEdge{V}) where {V}
    ls = linkinds(tn, edge)
    return prod([isnothing(l) ? 1 : dim(l) for l in ls])
end

function linkdims(tn::AbstractITensorNetwork{V}) where {V}
    ld = DataGraph{V}(
        copy(underlying_graph(tn)); vertex_data_type = Nothing, edge_data_type = Int
    )
    for e in edges(ld)
        ld[e] = linkdim(tn, e)
    end
    return ld
end

# Add a new edge between two vertices by threading a fresh link `Index`
# via QR; the second `setindex!` writes that `Index` into both endpoint
# tensors, and reverse-map reconciliation picks up the graph edge as a
# consequence. Returns the `Graphs.add_edge!` convention: `true` if a new
# graph edge was added, `false` if it was already there.
function Graphs.add_edge!(tn::AbstractITensorNetwork, edge)
    has_edge(tn, edge) && return false
    qr!(tn, edge)
    return true
end

# TODO: What to output? Could be an `IndsNetwork`. Or maybe
# that would be a different function `commonindsnetwork`.
# Even in that case, this could output a `Dictionary`
# from the edges to the common inds on that edge.
function ITensors.commoninds(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
    inds = Index[]
    for v1 in vertices(tn1)
        for v2 in vertices(tn2)
            append!(inds, commoninds(tn1[v1], tn2[v2]))
        end
    end
    return inds
end

"""
    add(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork) -> ITensorNetwork

Add two `ITensorNetwork`s together by taking their direct sum (growing the bond dimension).
The result represents the state `tn1 + tn2`, with bond dimension on each edge equal to the
sum of the bond dimensions of `tn1` and `tn2`.

Both networks must have the same vertex set and matching site indices at each vertex.

Use `truncate` on the result to compress back to a lower bond dimension.

See also: `Base.:+` for `TreeTensorNetwork`, `truncate`.
"""
function add(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork)
    @assert issetequal(vertices(tn1), vertices(tn2))

    # Collapse any multi-edges (edges carrying >1 shared Index) to a single
    # shared Index, so the per-edge direct sum below sees one link per edge.
    tn1 = copy(tn1)
    tn2 = copy(tn2)
    for e in edges(tn1)
        length(linkinds(tn1, e)) > 1 && qr!(tn1, e)
    end
    for e in edges(tn2)
        length(linkinds(tn2, e)) > 1 && qr!(tn2, e)
    end

    edges_tn1, edges_tn2 = edges(tn1), edges(tn2)

    if !issetequal(edges_tn1, edges_tn2)
        new_edges = union(edges_tn1, edges_tn2)
        tn1 = add_edges(tn1, new_edges)
        tn2 = add_edges(tn2, new_edges)
    end

    edges_tn1, edges_tn2 = edges(tn1), edges(tn2)
    @assert issetequal(edges_tn1, edges_tn2)

    tn12 = copy(tn1)
    new_edge_indices = Dict(
        zip(
            edges_tn1,
            [
                Index(
                        dim(only(linkinds(tn1, e))) + dim(only(linkinds(tn2, e))),
                        tags(only(linkinds(tn1, e)))
                    ) for e in edges_tn1
            ]
        )
    )

    #Create vertices of tn12 as direct sum of tn1[v] and tn2[v]. Work out the matching indices by matching edges. Make index tags those of tn1[v]
    for v in vertices(tn1)
        @assert issetequal(siteinds(tn1, v), siteinds(tn2, v))

        e1_v = filter(x -> src(x) == v || dst(x) == v, edges_tn1)
        e2_v = filter(x -> src(x) == v || dst(x) == v, edges_tn2)

        @assert issetequal(e1_v, e2_v)
        tn1v_linkinds = Index[only(linkinds(tn1, e)) for e in e1_v]
        tn2v_linkinds = Index[only(linkinds(tn2, e)) for e in e1_v]
        tn12v_linkinds = Index[new_edge_indices[e] for e in e1_v]

        @assert length(tn1v_linkinds) == length(tn2v_linkinds)

        tn12[v] = ITensors.directsum(
            tn12v_linkinds,
            tn1[v] => Tuple(tn1v_linkinds),
            tn2[v] => Tuple(tn2v_linkinds);
            tags = tags.(Tuple(tn1v_linkinds))
        )
    end

    return tn12
end

Base.:+(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork) = add(tn1, tn2)

ITensors.hasqns(tn::AbstractITensorNetwork) = any(v -> hasqns(tn[v]), vertices(tn))

function NamedGraphs.induced_subgraph_from_vertices(
        itn::AbstractITensorNetwork,
        subvertices
    )
    _, vlist = induced_subgraph(underlying_graph(itn), subvertices)
    sub_vs = collect(subvertices)
    return ITensorNetwork(Dictionary(sub_vs, [itn[v] for v in sub_vs])), vlist
end
