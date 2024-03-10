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

# Pad with last value to length.
# If it is a single value (non-Vector), fill with
# that value to the length.
extend(x::Vector, length::Int) = [x; fill(last(x), length - Base.length(x))]
extend(x, length::Int) = extend([x], length)

# Treat `AbstractArray` as leaves.

struct AbstractArrayLeafStyle <: WalkStyle end

StructWalk.children(::AbstractArrayLeafStyle, x::AbstractArray) = ()

function extend_columns(nt::NamedTuple, length::Int)
  return map(x -> extend(x, length), nt)
end

function extend_columns_recursive(nt::NamedTuple, length::Int)
  return postwalk(AbstractArrayLeafStyle(), nt) do x
    x isa NamedTuple && return x

    return extend(x, length)
  end
end

#ToDo: remove
#nrows(nt::NamedTuple) = isempty(nt) ? 0 : length(first(nt))

function row(nt::NamedTuple, i::Int)
  isempty(nt) ? (return nt) : (return map(x -> x[i], nt))
end

# Similar to `Tables.rowtable(x)`

function rows(nt::NamedTuple, length::Int)
  return [row(nt, i) for i in 1:length]
end

function rows_recursive(nt::NamedTuple, length::Int)
  return postwalk(AbstractArrayLeafStyle(), nt) do x
    !(x isa NamedTuple) && return x

    return rows(x, length)
  end
end

function expand(nt::NamedTuple, length::Int)
  nt_padded = extend_columns_recursive(nt, length)
  return rows_recursive(nt_padded, length)
end

function interleave(a::Vector, b::Vector)
  ab = flatten(collect(zip(a, b)))
  if length(a) == length(b)
    return ab
  elseif length(a) == length(b) + 1
    return append!(ab, [last(a)])
  else
    error(
      "Trying to interleave vectors of length $(length(a)) and $(length(b)), not implemented.",
    )
  end
end