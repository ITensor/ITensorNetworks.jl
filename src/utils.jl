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

#Find the edge in the list of edges which involves v1 and v2
function find_edge(edges::Vector{NamedDimEdge{Tuple}}, v1::Tuple, v2::Tuple)
  for i in 1:length(edges)
    if (
      (src(edges[i]) == v1 && dst(edges[i]) == v2) ||
      (dst(edges[i]) == v1 && src(edges[i]) == v2)
    )
      return i
    end
  end
  return 0
end
