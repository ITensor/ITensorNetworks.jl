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

front(itr, n=1) = Iterators.take(itr, length(itr) - n)
tail(itr) = Iterators.drop(itr, 1)

# Tree utils
function line_to_tree(line::Vector)
  if length(line) == 1 && line[1] isa Vector
    return line[1]
  end
  if length(line) <= 2
    return line
  end
  return [line_to_tree(line[1:(end - 1)]), line[end]]
end
