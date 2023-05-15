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

"""Given a vector of gates acting on siteinds within s, separate them into groups of commuting gates (i.e. gates in the same group act on different physical indices)"""
function group_gates(s::IndsNetwork, gates::Vector{ITensor})

  remaining_gates = copy(gates)
  gate_groups = Vector{ITensor}[]

  while !isempty(remaining_gates)
      cur_group = ITensor[]
      cur_vertices = []
      inds_to_remove = []
      for i in 1:length(remaining_gates)
          gate = remaining_gates[i]
          vs =  vertices(s)[findall(i -> (length(commoninds(s[i], inds(gate))) != 0), vertices(s))]

          if isempty(vs)
            error("Gate does not appear to have any indices within the indsnetwork provided")
          end

          if all([v ∉ cur_vertices for v in vs])
              push!(cur_group, gate)
              push!(cur_vertices, vs...)
              push!(inds_to_remove, i)
          end
      end
      remaining_gates = ITensor[remaining_gates[i] for i in setdiff([i for i in 1:length(remaining_gates)], inds_to_remove)]
      push!(gate_groups, cur_group)
  end

  return gate_groups
end