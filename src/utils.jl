to_tuple(x) = (x,)
to_tuple(x::Tuple) = x

function cartesian_to_linear(dims::Tuple)
  return Dictionary(vec(Tuple.(CartesianIndices(dims))), 1:prod(dims))
end

# Convert to real if possible
maybe_real(x::Real) = x
maybe_real(x::Complex) = iszero(imag(x)) ? real(x) : x

maybe_only(x) = x
maybe_only(x::Tuple{T}) where {T} = only(x)


#Find the edge in the list of edges which goes from source to dest (use the negative index if the edge is reversed)
function find_edge(edges::Vector{NamedDimEdge{Tuple}}, source::Tuple, dest::Tuple)
  for i = 1:length(edges)
      if((src(edges[i]) == source && dst(edges[i]) == dest) || (dst(edges[i]) == source && src(edges[i]) == dest))
          return i
      end
  end
  return 0
end

#Get subset of edges involving a specific vertex
function find_edges_involving_vertex(edges::Vector{NamedDimEdge{Tuple}}, vertex::Tuple)
  e_out = []
  for e in edges
      if(src(e) == vertex || dst(e) == vertex)
          push!(e_out, e)
      end
  end

  return e_out
end