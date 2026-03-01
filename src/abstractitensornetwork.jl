using .ITensorsExtensions: ITensorsExtensions, indtype, promote_indtype
using Adapt: Adapt, adapt, adapt_structure
using DataGraphs:
    DataGraphs, edge_data, underlying_graph, underlying_graph_type, vertex_data
using Dictionaries: Dictionary
using Graphs: Graphs, Graph, add_edge!, add_vertex!, bfs_tree, center, dst, edges, edgetype,
    ne, neighbors, rem_edge!, src, vertices
using ITensors: ITensors, @Algorithm_str, ITensor, addtags, combiner, commoninds,
    commontags, contract, dag, hascommoninds, noprime, onehot, prime, replaceprime,
    replacetags, setprime, settags, sim, swaptags, unioninds, uniqueinds
using LinearAlgebra: LinearAlgebra, factorize
using MacroTools: @capture
using NDTensors: NDTensors, Algorithm, dim
using NamedGraphs.GraphsExtensions:
    directed_graph, incident_edges, rename_vertices, vertextype, ⊔
using NamedGraphs: NamedGraphs, NamedGraph, not_implemented, steiner_tree
using SplitApplyCombine: flatten

abstract type AbstractITensorNetwork{V} <: AbstractDataGraph{V, ITensor, ITensor} end

# Field access
data_graph_type(::Type{<:AbstractITensorNetwork}) = not_implemented()
data_graph(graph::AbstractITensorNetwork) = not_implemented()

# TODO: Define a generic fallback for `AbstractDataGraph`?
DataGraphs.edge_data_eltype(::Type{<:AbstractITensorNetwork}) = ITensor

# Graphs.jl overloads
function Graphs.weights(graph::AbstractITensorNetwork)
    V = vertextype(graph)
    es = Tuple.(edges(graph))
    ws = Dictionary{Tuple{V, V}, Float64}(es, undef)
    for e in edges(graph)
        w = log2(dim(commoninds(graph, e)))
        ws[(src(e), dst(e))] = w
    end
    return ws
end

# Copy
Base.copy(tn::AbstractITensorNetwork) = not_implemented()

# Iteration
Base.iterate(tn::AbstractITensorNetwork, args...) = iterate(vertex_data(tn), args...)

# TODO: This contrasts with the `DataGraphs.AbstractDataGraph` definition,
# where it is defined as the `vertextype`. Does that cause problems or should it be changed?
Base.eltype(tn::AbstractITensorNetwork) = eltype(vertex_data(tn))

# Overload if needed
Graphs.is_directed(::Type{<:AbstractITensorNetwork}) = false

# Derived interface, may need to be overloaded
function DataGraphs.underlying_graph_type(G::Type{<:AbstractITensorNetwork})
    return underlying_graph_type(data_graph_type(G))
end

function ITensors.datatype(tn::AbstractITensorNetwork)
    return mapreduce(v -> datatype(tn[v]), promote_type, vertices(tn))
end

# AbstractDataGraphs overloads
function DataGraphs.vertex_data(graph::AbstractITensorNetwork, args...)
    return vertex_data(data_graph(graph), args...)
end
function DataGraphs.edge_data(graph::AbstractITensorNetwork, args...)
    return edge_data(data_graph(graph), args...)
end

DataGraphs.underlying_graph(tn::AbstractITensorNetwork) = underlying_graph(data_graph(tn))
function NamedGraphs.vertex_positions(tn::AbstractITensorNetwork)
    return NamedGraphs.vertex_positions(underlying_graph(tn))
end
function NamedGraphs.ordered_vertices(tn::AbstractITensorNetwork)
    return NamedGraphs.ordered_vertices(underlying_graph(tn))
end

function Adapt.adapt_structure(to, tn::AbstractITensorNetwork)
    # TODO: Define and use:
    #
    # @preserve_graph map_vertex_data(adapt(to), tn)
    #
    # or just:
    #
    # @preserve_graph map(adapt(to), tn)
    return map_vertex_data_preserve_graph(adapt(to), tn)
end

#
# Iteration
#

# TODO: iteration

# TODO: different `map` functionalities as defined for ITensors.AbstractMPS

# TODO: broadcasting

function Base.union(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kwargs...)
    # TODO: Use a different constructor call here?
    tn = _ITensorNetwork(union(data_graph(tn1), data_graph(tn2)); kwargs...)
    # Add any new edges that are introduced during the union
    for v1 in vertices(tn1)
        for v2 in vertices(tn2)
            if hascommoninds(tn, v1 => v2)
                add_edge!(tn, v1 => v2)
            end
        end
    end
    return tn
end

function NamedGraphs.rename_vertices(f::Function, tn::AbstractITensorNetwork)
    # TODO: Use a different constructor call here?
    return _ITensorNetwork(rename_vertices(f, data_graph(tn)))
end

#
# Data modification
#

function setindex_preserve_graph!(tn::AbstractITensorNetwork, value, vertex)
    data_graph(tn)[vertex] = value
    return tn
end

# TODO: Move to `BaseExtensions` module.
function is_setindex!_expr(expr::Expr)
    return is_assignment_expr(expr) && is_getindex_expr(first(expr.args))
end
is_setindex!_expr(x) = false
is_getindex_expr(expr::Expr) = (expr.head === :ref)
is_getindex_expr(x) = false
is_assignment_expr(expr::Expr) = (expr.head === :(=))
is_assignment_expr(expr) = false

# TODO: Define this in terms of a function mapping
# preserve_graph_function(::typeof(setindex!)) = setindex!_preserve_graph
# preserve_graph_function(::typeof(map_vertex_data)) = map_vertex_data_preserve_graph
# Also allow annotating codeblocks like `@views`.
macro preserve_graph(expr)
    if !is_setindex!_expr(expr)
        error(
            "preserve_graph must be used with setindex! syntax (as @preserve_graph a[i,j,...] = value)"
        )
    end
    @capture(expr, array_[indices__] = value_)
    return :(setindex_preserve_graph!($(esc(array)), $(esc(value)), $(esc.(indices)...)))
end

function ITensors.hascommoninds(tn::AbstractITensorNetwork, edge::Pair)
    return hascommoninds(tn, edgetype(tn)(edge))
end

function ITensors.hascommoninds(tn::AbstractITensorNetwork, edge::AbstractEdge)
    return hascommoninds(tn[src(edge)], tn[dst(edge)])
end

function Base.setindex!(tn::AbstractITensorNetwork, value, v)
    # v = to_vertex(tn, index...)
    @preserve_graph tn[v] = value
    for edge in incident_edges(tn, v)
        rem_edge!(tn, edge)
    end
    for vertex in vertices(tn)
        if v ≠ vertex
            edge = v => vertex
            if hascommoninds(tn, edge)
                add_edge!(tn, edge)
            end
        end
    end
    return tn
end

# Convenience wrapper
function eachtensor(tn::AbstractITensorNetwork, vertices = vertices(tn))
    return map(v -> tn[v], vertices)
end

#
# Promotion and conversion
#

function ITensorsExtensions.promote_indtypeof(tn::AbstractITensorNetwork)
    return mapreduce(promote_indtype, eachtensor(tn)) do t
        return indtype(t)
    end
end

function NDTensors.scalartype(tn::AbstractITensorNetwork)
    return mapreduce(eltype, promote_type, eachtensor(tn); init = Bool)
end

# TODO: Define `eltype(::AbstractITensorNetwork)` as `ITensor`?

# TODO: Implement using `adapt`
function NDTensors.convert_scalartype(eltype::Type{<:Number}, tn::AbstractITensorNetwork)
    tn = copy(tn)
    vertex_data(tn) .= ITensors.adapt.(Ref(eltype), vertex_data(tn))
    return tn
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
        is[v] = uniqueinds(tn, v)
    end
    for e in edges(tn)
        is[e] = commoninds(tn, e)
    end
    return is
end

# Alias
indsnetwork(tn::AbstractITensorNetwork) = IndsNetwork(tn)

# TODO: Output a `VertexDataGraph`? Unfortunately
# `IndsNetwork` doesn't allow iterating over vertex data.
function siteinds(tn::AbstractITensorNetwork)
    is = IndsNetwork(underlying_graph(tn))
    for v in vertices(tn)
        is[v] = uniqueinds(tn, v)
    end
    return is
end

function flatten_siteinds(tn::AbstractITensorNetwork)
    # `identity.(...)` narrows the type, maybe there is a better way.
    return identity.(flatten(map(v -> siteinds(tn, v), vertices(tn))))
end

function linkinds(tn::AbstractITensorNetwork)
    is = IndsNetwork(underlying_graph(tn))
    for e in edges(tn)
        is[e] = commoninds(tn, e)
    end
    return is
end

function flatten_linkinds(tn::AbstractITensorNetwork)
    # `identity.(...)` narrows the type, maybe there is a better way.
    return identity.(flatten(map(e -> linkinds(tn, e), edges(tn))))
end

#
# Index access
#

function neighbor_tensors(tn::AbstractITensorNetwork, vertex)
    return eachtensor(tn, neighbors(tn, vertex))
end

function ITensors.uniqueinds(tn::AbstractITensorNetwork, vertex)
    tn_vertex = [tn[vertex]; collect(neighbor_tensors(tn, vertex))]
    return reduce(setdiff, inds.(tn_vertex))
end

function ITensors.uniqueinds(tn::AbstractITensorNetwork, edge::AbstractEdge)
    return uniqueinds(tn[src(edge)], tn[dst(edge)])
end

function ITensors.uniqueinds(tn::AbstractITensorNetwork, edge::Pair)
    return uniqueinds(tn, edgetype(tn)(edge))
end

function siteinds(tn::AbstractITensorNetwork, vertex)
    return uniqueinds(tn, vertex)
end
# Fix ambiguity error with IndsNetwork constructor.
function siteinds(tn::AbstractITensorNetwork, vertex::Int)
    return uniqueinds(tn, vertex)
end

function ITensors.commoninds(tn::AbstractITensorNetwork, edge)
    e = edgetype(tn)(edge)
    return commoninds(tn[src(e)], tn[dst(e)])
end

function linkinds(tn::AbstractITensorNetwork, edge)
    return commoninds(tn, edge)
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
        @preserve_graph tn[v] = replaceinds(tn[v], is[v] => is′[v])
    end
    for e in edges(is)
        isassigned(is, e) || continue
        for v in (src(e), dst(e))
            @preserve_graph tn[v] = replaceinds(tn[v], is[e] => is′[e])
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

function map_vertex_data(f, tn::AbstractITensorNetwork)
    tn = copy(tn)
    for v in vertices(tn)
        tn[v] = f(tn[v])
    end
    return tn
end

# TODO: Define @preserve_graph map_vertex_data(f, tn)`
function map_vertex_data_preserve_graph(f, tn::AbstractITensorNetwork)
    tn = copy(tn)
    for v in vertices(tn)
        @preserve_graph tn[v] = f(tn[v])
    end
    return tn
end

function map_vertices_preserve_graph!(
        f,
        tn::AbstractITensorNetwork;
        vertices = vertices(tn)
    )
    for v in vertices
        @preserve_graph tn[v] = f(v)
    end
    return tn
end

function Base.conj(tn::AbstractITensorNetwork)
    # TODO: Use `@preserve_graph map_vertex_data(f, tn)`
    return map_vertex_data_preserve_graph(conj, tn)
end

function ITensors.dag(tn::AbstractITensorNetwork)
    # TODO: Use `@preserve_graph map_vertex_data(f, tn)`
    return map_vertex_data_preserve_graph(dag, tn)
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
    neighbors_src = setdiff(neighbors(tn, src(edge)), [dst(edge)])
    neighbors_dst = setdiff(neighbors(tn, dst(edge)), [src(edge)])
    new_itensor = tn[src(edge)] * tn[dst(edge)]
    # The following is equivalent to:
    #
    # tn[dst(edge)] = new_itensor
    #
    # but without having to search all vertices
    # to update the edges.
    rem_vertex!(tn, src(edge))
    rem_vertex!(tn, dst(edge))
    add_vertex!(tn, merged_vertex)
    for n_src in neighbors_src
        add_edge!(tn, merged_vertex => n_src)
    end
    for n_dst in neighbors_dst
        add_edge!(tn, merged_vertex => n_dst)
    end
    @preserve_graph tn[merged_vertex] = new_itensor
    return tn
end

function ITensors.tags(tn::AbstractITensorNetwork, edge)
    is = linkinds(tn, edge)
    return commontags(is)
end

function LinearAlgebra.svd(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
    return svd(tn, edgetype(tn)(edge))
end

function LinearAlgebra.svd(
        tn::AbstractITensorNetwork,
        edge::AbstractEdge;
        U_vertex = src(edge),
        S_vertex = (edge, "S"),
        V_vertex = (edge, "V"),
        u_tags = tags(tn, edge),
        v_tags = tags(tn, edge),
        kwargs...
    )
    tn = copy(tn)
    left_inds = uniqueinds(tn, edge)
    U, S, V =
        svd(tn[src(edge)], left_inds; lefttags = u_tags, righttags = v_tags, kwargs...)

    rem_vertex!(tn, src(edge))
    add_vertex!(tn, U_vertex)
    tn[U_vertex] = U

    add_vertex!(tn, S_vertex)
    tn[S_vertex] = S

    add_vertex!(tn, V_vertex)
    tn[V_vertex] = V

    return tn
end

function LinearAlgebra.qr(
        tn::AbstractITensorNetwork,
        edge::AbstractEdge;
        Q_vertex = src(edge),
        R_vertex = (edge, "R"),
        tags = tags(tn, edge),
        kwargs...
    )
    tn = copy(tn)
    left_inds = uniqueinds(tn, edge)
    Q, R = factorize(tn[src(edge)], left_inds; tags, kwargs...)

    rem_vertex!(tn, src(edge))
    add_vertex!(tn, Q_vertex)
    tn[Q_vertex] = Q

    add_vertex!(tn, R_vertex)
    tn[R_vertex] = R

    return tn
end

function LinearAlgebra.factorize(
        tn::AbstractITensorNetwork,
        edge::AbstractEdge;
        X_vertex = src(edge),
        Y_vertex = ("Y", edge),
        tags = tags(tn, edge),
        kwargs...
    )
    # Promote vertex type
    V = promote_type(vertextype(tn), typeof(X_vertex), typeof(Y_vertex))

    # TODO: Check `ITensorNetwork{V}`, shouldn't need a copy here.
    tn = ITensorNetwork{V}(copy(tn))

    neighbors_X = setdiff(neighbors(tn, src(edge)), [dst(edge)])
    left_inds = uniqueinds(tn, edge)
    X, Y = factorize(tn[src(edge)], left_inds; tags, kwargs...)

    rem_vertex!(tn, src(edge))
    add_vertex!(tn, X_vertex)
    add_vertex!(tn, Y_vertex)

    add_edge!(tn, X_vertex => Y_vertex)
    for nX in neighbors_X
        add_edge!(tn, X_vertex => nX)
    end
    add_edge!(tn, Y_vertex => dst(edge))
    @preserve_graph tn[X_vertex] = X
    @preserve_graph tn[Y_vertex] = Y
    return tn
end

function LinearAlgebra.factorize(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
    return factorize(tn, edgetype(tn)(edge); kwargs...)
end

# For ambiguity error; TODO: decide whether to use graph mutating methods when resulting graph is unchanged?
function gauge_edge(
        alg::Algorithm"orthogonalize", tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...
    )
    # tn = factorize(tn, edge; kwargs...)
    # # TODO: Implement as `only(common_neighbors(tn, src(edge), dst(edge)))`
    # new_vertex = only(neighbors(tn, src(edge)) ∩ neighbors(tn, dst(edge)))
    # return contract(tn, new_vertex => dst(edge))
    !has_edge(tn, edge) && throw(ArgumentError("Edge not in graph."))
    tn = copy(tn)
    left_inds = uniqueinds(tn, edge)
    ltags = tags(tn, edge)
    X, Y = factorize(tn[src(edge)], left_inds; tags = ltags, ortho = "left", kwargs...)
    @preserve_graph tn[src(edge)] = X
    @preserve_graph tn[dst(edge)] = tn[dst(edge)] * Y
    return tn
end

# For ambiguity error; TODO: decide whether to use graph mutating methods when resulting graph is unchanged?
function gauge_walk(
        alg::Algorithm, tn::AbstractITensorNetwork, edges::Vector{<:AbstractEdge}; kwargs...
    )
    tn = copy(tn)
    for edge in edges
        tn = gauge_edge(alg, tn, edge; kwargs...)
    end
    return tn
end

function gauge_walk(alg::Algorithm, tn::AbstractITensorNetwork, edge::Pair; kwargs...)
    return gauge_edge(alg::Algorithm, tn, edgetype(tn)(edge); kwargs...)
end

function gauge_walk(
        alg::Algorithm, tn::AbstractITensorNetwork, edges::Vector{<:Pair}; kwargs...
    )
    return gauge_walk(alg, tn, edgetype(tn).(edges); kwargs...)
end

function tree_gauge(alg::Algorithm, ψ::AbstractITensorNetwork, region)
    return tree_gauge(alg, ψ, [region])
end

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
function tree_gauge(
        alg::Algorithm,
        ψ::AbstractITensorNetwork,
        cur_region::Vector,
        new_region::Vector;
        kwargs...
    )
    es = edge_sequence_between_regions(ψ, cur_region, new_region)
    ψ = gauge_walk(alg, ψ, es; kwargs...)
    return ψ
end

# Gauge a ITensorNetwork towards a region, treating
# the network as a tree spanned by a spanning tree.
function tree_gauge(alg::Algorithm, ψ::AbstractITensorNetwork, region::Vector)
    return tree_gauge(alg, ψ, collect(vertices(ψ)), region)
end

function tree_orthogonalize(ψ::AbstractITensorNetwork, cur_region, new_region; kwargs...)
    return tree_gauge(Algorithm("orthogonalize"), ψ, cur_region, new_region; kwargs...)
end

function tree_orthogonalize(ψ::AbstractITensorNetwork, region; kwargs...)
    return tree_gauge(Algorithm("orthogonalize"), ψ, region; kwargs...)
end

# TODO: decide whether to use graph mutating methods when resulting graph is unchanged?
function _truncate_edge(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
    !has_edge(tn, edge) && throw(ArgumentError("Edge not in graph."))
    tn = copy(tn)
    left_inds = uniqueinds(tn, edge)
    ltags = tags(tn, edge)
    U, S, V = svd(tn[src(edge)], left_inds; lefttags = ltags, kwargs...)
    @preserve_graph tn[src(edge)] = U
    @preserve_graph tn[dst(edge)] = tn[dst(edge)] * (S * V)
    return tn
end

"""
    truncate(tn::AbstractITensorNetwork, edge; kwargs...) -> ITensorNetwork

Truncate the bond across `edge` in `tn` by performing an SVD and discarding small
singular values. `edge` may be an `AbstractEdge` or a `Pair` of vertices.

Truncation parameters are passed as keyword arguments and forwarded to `ITensors.svd`:

  - `cutoff`: Drop singular values smaller than this threshold.
  - `maxdim`: Maximum number of singular values to keep.
  - `mindim`: Minimum number of singular values to keep.

This operates on a single bond. For `TreeTensorNetwork`, the no-argument form
`truncate(ttn; kwargs...)` sweeps all bonds and is generally preferred for full
recompression after addition or subspace expansion.

See also: `Base.truncate(::AbstractTreeTensorNetwork)`.
"""
function Base.truncate(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
    return _truncate_edge(tn, edge; kwargs...)
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

function linkinds_combiners(tn::AbstractITensorNetwork; edges = edges(tn))
    combiners = DataGraph(
        directed_graph(underlying_graph(tn));
        vertex_data_eltype = ITensor,
        edge_data_eltype = ITensor
    )
    for e in edges
        C = combiner(linkinds(tn, e); tags = edge_tag(e))
        combiners[e] = C
        combiners[reverse(e)] = dag(C)
    end
    return combiners
end

function combine_linkinds(tn::AbstractITensorNetwork, combiners)
    combined_tn = copy(tn)
    for e in edges(tn)
        if !isempty(linkinds(tn, e)) && haskey(edge_data(combiners), e)
            combined_tn[src(e)] = combined_tn[src(e)] * combiners[e]
            combined_tn[dst(e)] = combined_tn[dst(e)] * combiners[reverse(e)]
        end
    end
    return combined_tn
end

function combine_linkinds(
        tn::AbstractITensorNetwork; edges::Vector{<:Union{Pair, AbstractEdge}} = edges(tn)
    )
    combiners = linkinds_combiners(tn; edges)
    return combine_linkinds(tn, combiners)
end

function split_index(
        tn::AbstractITensorNetwork,
        edges_to_split;
        src_ind_map::Function = identity,
        dst_ind_map::Function = prime
    )
    tn = copy(tn)
    for e in edges_to_split
        inds = commoninds(tn[src(e)], tn[dst(e)])
        tn[src(e)] = replaceinds(tn[src(e)], inds, src_ind_map(inds))
        tn[dst(e)] = replaceinds(tn[dst(e)], inds, dst_ind_map(inds))
    end

    return tn
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

norm_sqr_network(ψ::AbstractITensorNetwork) = inner_network(ψ, ψ)

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

# TODO: Move to an `ITensorNetworksVisualizationInterfaceExt`
# package extension (and define a `VisualizationInterface` package
# based on `ITensorVisualizationCore`.).
using ITensors.ITensorVisualizationCore: ITensorVisualizationCore, visualize
function ITensorVisualizationCore.visualize(
        tn::AbstractITensorNetwork,
        args...;
        vertex_labels_prefix = nothing,
        vertex_labels = nothing,
        kwargs...
    )
    if !isnothing(vertex_labels_prefix)
        vertex_labels = [vertex_labels_prefix * string(v) for v in vertices(tn)]
    end
    # TODO: Use `tokenize_vertex`.
    return visualize(collect(eachtensor(tn)), args...; vertex_labels, kwargs...)
end

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
        copy(underlying_graph(tn)); vertex_data_eltype = Nothing, edge_data_eltype = Int
    )
    for e in edges(ld)
        ld[e] = linkdim(tn, e)
    end
    return ld
end

#
# Site combiners
#

# TODO: will be broken, fix this
function site_combiners(tn::AbstractITensorNetwork{V}) where {V}
    Cs = DataGraph{V, ITensor}(copy(underlying_graph(tn)))
    for v in vertices(tn)
        s = siteinds(tn, v)
        Cs[v] = combiner(s; tags = commontags(s))
    end
    return Cs
end

function insert_linkinds(
        tn::AbstractITensorNetwork, edges = edges(tn); link_space = trivial_space(tn)
    )
    tn = copy(tn)
    for e in edges
        if !hascommoninds(tn, e)
            iₑ = Index(link_space, edge_tag(e))
            X = onehot(iₑ => 1)
            tn[src(e)] *= X
            tn[dst(e)] *= dag(X)
        end
    end
    return tn
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
Check if the edge of an itensornetwork has multiple indices
"""
is_multi_edge(tn::AbstractITensorNetwork, e) = length(linkinds(tn, e)) > 1
is_multi_edge(tn::AbstractITensorNetwork) = Base.Fix1(is_multi_edge, tn)

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

    tn1 = combine_linkinds(tn1; edges = filter(is_multi_edge(tn1), edges(tn1)))
    tn2 = combine_linkinds(tn2; edges = filter(is_multi_edge(tn2), edges(tn2)))

    edges_tn1, edges_tn2 = edges(tn1), edges(tn2)

    if !issetequal(edges_tn1, edges_tn2)
        new_edges = union(edges_tn1, edges_tn2)
        tn1 = insert_linkinds(tn1, new_edges)
        tn2 = insert_linkinds(tn2, new_edges)
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

"""
Scale each tensor of the network via a function vertex -> Number
"""
function scale!(
        weight_function::Function,
        tn::AbstractITensorNetwork;
        vertices = collect(Graphs.vertices(tn))
    )
    return map_vertices_preserve_graph!(v -> weight_function(v) * tn[v], tn; vertices)
end

"""
Scale each tensor of the network by a scale factor for each vertex in the keys of the dictionary
"""
function scale!(tn::AbstractITensorNetwork, vertices_weights::Dictionary)
    return scale!(v -> vertices_weights[v], tn; vertices = keys(vertices_weights))
end

function scale(weight_function::Function, tn; kwargs...)
    tn = copy(tn)
    return scale!(weight_function, tn; kwargs...)
end

function scale(tn::AbstractITensorNetwork, vertices_weights::Dictionary; kwargs...)
    tn = copy(tn)
    return scale!(tn, vertices_weights; kwargs...)
end

Base.:+(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork) = add(tn1, tn2)

ITensors.hasqns(tn::AbstractITensorNetwork) = any(v -> hasqns(tn[v]), vertices(tn))
