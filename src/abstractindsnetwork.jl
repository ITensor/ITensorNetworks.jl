using ITensors: IndexSet
using DataGraphs: DataGraphs, AbstractDataGraph, edge_data, vertex_data
using Graphs: Graphs, AbstractEdge
using ITensors: ITensors, unioninds, uniqueinds
using NamedGraphs: NamedGraphs
using NamedGraphs.GraphsExtensions: incident_edges, rename_vertices

abstract type AbstractIndsNetwork{V,I} <: AbstractDataGraph{V,Vector{I},Vector{I}} end

# Field access
data_graph(graph::AbstractIndsNetwork) = not_implemented()

# Overload if needed
Graphs.is_directed(::Type{<:AbstractIndsNetwork}) = false

# AbstractDataGraphs overloads
function DataGraphs.vertex_data(graph::AbstractIndsNetwork, args...)
  return vertex_data(data_graph(graph), args...)
end
function DataGraphs.edge_data(graph::AbstractIndsNetwork, args...)
  return edge_data(data_graph(graph), args...)
end

# TODO: Define a generic fallback for `AbstractDataGraph`?
DataGraphs.edge_data_eltype(::Type{<:AbstractIndsNetwork{V,I}}) where {V,I} = Vector{I}

## TODO: Bring these back.
## function indsnetwork_getindex(is::AbstractIndsNetwork, index)
##   return get(data_graph(is), index, indtype(is)[])
## end
## 
## function Base.getindex(is::AbstractIndsNetwork, index)
##   return indsnetwork_getindex(is, index)
## end
## 
## function Base.getindex(is::AbstractIndsNetwork, index::Pair)
##   return indsnetwork_getindex(is, index)
## end
## 
## function Base.getindex(is::AbstractIndsNetwork, index::AbstractEdge)
##   return indsnetwork_getindex(is, index)
## end
## 
## function indsnetwork_setindex!(is::AbstractIndsNetwork, value, index)
##   data_graph(is)[index] = value
##   return is
## end
## 
## function Base.setindex!(is::AbstractIndsNetwork, value, index)
##   indsnetwork_setindex!(is, value, index)
##   return is
## end
## 
## function Base.setindex!(is::AbstractIndsNetwork, value, index::Pair)
##   indsnetwork_setindex!(is, value, index)
##   return is
## end
## 
## function Base.setindex!(is::AbstractIndsNetwork, value, index::AbstractEdge)
##   indsnetwork_setindex!(is, value, index)
##   return is
## end
## 
## function Base.setindex!(is::AbstractIndsNetwork, value::Index, index)
##   indsnetwork_setindex!(is, value, index)
##   return is
## end

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

function promote_indtypeof(is::AbstractIndsNetwork)
  sitetype = mapreduce(promote_indtype, vertices(is); init=Index{Int}) do v
    # TODO: Replace with `is[v]` once `getindex(::IndsNetwork, ...)` is smarter.
    return mapreduce(typeof, promote_indtype, get(is, v, Index[]); init=Index{Int})
  end
  linktype = mapreduce(promote_indtype, edges(is); init=Index{Int}) do e
    # TODO: Replace with `is[e]` once `getindex(::IndsNetwork, ...)` is smarter.
    return mapreduce(typeof, promote_indtype, get(is, e, Index[]); init=Index{Int})
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
  edges=edges(indsnetwork);
  link_space=trivial_space(indsnetwork),
)
  indsnetwork = copy(indsnetwork)
  for e in edges
    # TODO: Change to check if it is empty.
    if !isassigned(indsnetwork, e)
      iₑ = Index(link_space, edge_tag(e))
      # TODO: Allow setting with just `Index`.
      indsnetwork[e] = [iₑ]
    end
  end
  return indsnetwork
end
