using .ITensorsExtensions: ITensorsExtensions, promote_indtype
using DataGraphs: DataGraphs, AbstractDataGraph, DataGraph, IsUnderlyingGraph, edge_data,
    get_edge_data, get_vertex_data, is_edge_assigned, is_vertex_assigned, map_data,
    set_edge_data!, set_vertex_data!, underlying_graph_type, vertex_data
using Graphs: Graphs, AbstractEdge
using ITensors: ITensors, IndexSet, unioninds, uniqueinds
using NamedGraphs.GraphsExtensions:
    GraphsExtensions, directed_graph, incident_edges, rename_vertices
using NamedGraphs: NamedGraphs

abstract type AbstractIndsNetwork{V, I} <: AbstractDataGraph{V, Vector{I}, Vector{I}} end

# Field access
data_graph(graph::AbstractIndsNetwork) = not_implemented()

# Overload if needed
Graphs.is_directed(::Type{<:AbstractIndsNetwork}) = false
GraphsExtensions.directed_graph(is::AbstractIndsNetwork) = directed_graph(data_graph(is))

# AbstractDataGraphs overloads
DataGraphs.underlying_graph(is::AbstractIndsNetwork) = underlying_graph(data_graph(is))

# TODO: Define a generic fallback for `AbstractDataGraph`?
DataGraphs.edge_data_type(::Type{<:AbstractIndsNetwork{V, I}}) where {V, I} = Vector{I}
DataGraphs.vertex_data_type(::Type{<:AbstractIndsNetwork{V, I}}) where {V, I} = Vector{I}

function DataGraphs.is_vertex_assigned(is::AbstractIndsNetwork, v)
    return is_vertex_assigned(data_graph(is), v)
end

function DataGraphs.set_vertex_data!(is::AbstractIndsNetwork, v, data)
    return set_vertex_data!(data_graph(is), v, data)
end
function DataGraphs.get_vertex_data(is::AbstractIndsNetwork, v)
    return get_vertex_data(data_graph(is), v)
end

function DataGraphs.is_edge_assigned(is::AbstractIndsNetwork, v)
    return is_edge_assigned(data_graph(is), v)
end

function DataGraphs.set_edge_data!(is::AbstractIndsNetwork, v, data)
    return set_edge_data!(data_graph(is), v, data)
end
function DataGraphs.get_edge_data(is::AbstractIndsNetwork, v)
    return get_edge_data(data_graph(is), v)
end

#
# Index access
#

function ITensors.uniqueinds(is::AbstractIndsNetwork, edge::AbstractEdge)
    # TODO: Replace with `is[v]` once `getindex(::IndsNetwork, ...)` is smarter.
    inds = IndexSet(get(is, src(edge), Index[]))
    for ei in setdiff(incident_edges(is, src(edge)), [edge])
        # TODO: Replace with `is[v]` once `getindex(::IndsNetwork, ...)` is smarter.
        inds = unioninds(inds, get(is, ei, Index[]))
    end
    return inds
end

function ITensors.uniqueinds(is::AbstractIndsNetwork, edge::Pair)
    return uniqueinds(is, edgetype(is)(edge))
end

function Base.union(is1::AbstractIndsNetwork, is2::AbstractIndsNetwork; kwargs...)
    return IndsNetwork(union(data_graph(is1), data_graph(is2); kwargs...))
end

function NamedGraphs.rename_vertices(f::Function, tn::AbstractIndsNetwork)
    return IndsNetwork(rename_vertices(f, data_graph(tn)))
end

#
# Convenience functions
#

function ITensorsExtensions.promote_indtypeof(is::AbstractIndsNetwork)
    sitetype = mapreduce(promote_indtype, vertices(is); init = Index{Int}) do v
        # TODO: Replace with `is[v]` once `getindex(::IndsNetwork, ...)` is smarter.
        return mapreduce(typeof, promote_indtype, get(is, v, Index[]); init = Index{Int})
    end
    linktype = mapreduce(promote_indtype, edges(is); init = Index{Int}) do e
        # TODO: Replace with `is[e]` once `getindex(::IndsNetwork, ...)` is smarter.
        return mapreduce(typeof, promote_indtype, get(is, e, Index[]); init = Index{Int})
    end
    return promote_indtype(sitetype, linktype)
end

function union_all_inds(is_in::AbstractIndsNetwork...)
    @assert all(map(ug -> ug == underlying_graph(is_in[1]), underlying_graph.(is_in)))
    is_out = IndsNetwork(underlying_graph(is_in[1]))
    for v in vertices(is_out)
        # TODO: Remove this check.
        if any(isassigned(is, v) for is in is_in)
            # TODO: Change `get` to `getindex`.
            is_out[v] = unioninds([get(is, v, Index[]) for is in is_in]...)
        end
    end
    for e in edges(is_out)
        # TODO: Remove this check.
        if any(isassigned(is, e) for is in is_in)
            # TODO: Change `get` to `getindex`.
            is_out[e] = unioninds([get(is, e, Index[]) for is in is_in]...)
        end
    end
    return is_out
end

function insert_linkinds(
        indsnetwork::AbstractIndsNetwork,
        edges = edges(indsnetwork);
        link_space = trivial_space(indsnetwork)
    )
    indsnetwork = copy(indsnetwork)
    for e in edges
        # TODO: Change to check if it is empty.
        if !isassigned(indsnetwork, e)
            if !isnothing(link_space)
                iₑ = Index(link_space, edge_tag(e))
                # TODO: Allow setting with just `Index`.
                indsnetwork[e] = [iₑ]
            else
                indsnetwork[e] = []
            end
        end
    end
    return indsnetwork
end
