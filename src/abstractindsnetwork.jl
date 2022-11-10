abstract type AbstractIndsNetwork{I} <:
              AbstractNamedDimDataGraph{Vector{I},Vector{I},Tuple,NamedDimEdge{Tuple}} end

# Field access
data_graph(graph::AbstractIndsNetwork) = _not_implemented()

# Overload if needed
is_directed(::Type{<:AbstractIndsNetwork}) = false

# AbstractDataGraphs overloads
vertex_data(graph::AbstractIndsNetwork, args...) = vertex_data(data_graph(graph), args...)
edge_data(graph::AbstractIndsNetwork, args...) = edge_data(data_graph(graph), args...)

# 
# Index access
# 

function uniqueinds(is::AbstractIndsNetwork, edge::AbstractEdge)
  inds = IndexSet(get_assigned(is, src(edge), Index[]))
  for ei in setdiff(incident_edges(is, src(edge)...), [edge])
    inds = unioninds(inds, get_assigned(is, ei, Index[]))
  end
  return inds
end

function uniqueinds(is::AbstractIndsNetwork, edge::Pair)
  return uniqueinds(is, edgetype(is)(edge))
end

# 
# Convenience functions
# 

function union_all_inds(is_in::AbstractIndsNetwork{I}...) where {I}
  @assert all(map(ug -> ug == underlying_graph(is_in[1]), underlying_graph.(is_in)))
  is_out = IndsNetwork(underlying_graph(is_in[1]))
  for v in vertices(is_out)
    if any(isassigned(is, v) for is in is_in)
      is_out[v] = unioninds([get_assigned(is, v, Index[]) for is in is_in]...)
    end
  end
  for e in edges(is_out)
    if any(isassigned(is, e) for is in is_in)
      is_out[e] = unioninds([get_assigned(is, e, Index[]) for is in is_in]...)
    end
  end
  return is_out
end
