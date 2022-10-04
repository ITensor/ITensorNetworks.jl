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

function Base.merge(is1::AbstractIndsNetwork{I}, is2::AbstractIndsNetwork{I}) where {I}
  @assert underlying_graph(is1) == underlying_graph(is2)
  is = IndsNetwork(underlying_graph(is1))
  for v in vertices(is1)
    if isassigned(is1, v) || isassigned(is2, v)
      is[v] = unioninds(get_assigned(is1, v, Index[]), get_assigned(is2, v, Index[]))
    end
  end
  for e in edges(is1)
    if isassigned(is1, e) || isassigned(is2, e)
      is[e] = unioninds(get_assigned(is1, e, Index[]), get_assigned(is2, e, Index[]))
    end
  end
  return is
end
