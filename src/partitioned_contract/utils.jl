function _is_neighbored_subset(v::Vector, set::Set)
  if set == Set()
    return true
  end
  @assert issubset(set, Set(v))
  i_begin = 1
  while !(i_begin in set)
    i_begin += 1
  end
  i_end = length(v)
  while !(i_end in set)
    i_end -= 1
  end
  return Set(v[i_begin:i_end]) == set
end

# replace the subarray of `v1` with `v2`
function _replace_subarray(v1::Vector, v2::Vector)
  if v2 == []
    return v1
  end
  v1 = copy(v1)
  num = 0
  for i in 1:length(v1)
    if v1[i] in v2
      num += 1
      v1[i] = v2[num]
    end
  end
  @assert num == length(v2)
  return v1
end

function _neighbor_edges(graph, vs)
  return filter(
    e -> (e.src in vs && !(e.dst in vs)) || (e.dst in vs && !(e.src in vs)), edges(graph)
  )
end

# Find the permutation to change `v1` into `v2`
function _findperm(v1, v2)
  index_to_number = Dict()
  for (i, v) in enumerate(v2)
    index_to_number[v] = i
  end
  v1_num = [index_to_number[v] for v in v1]
  return sortperm(v1_num)
end

function _bubble_sort(v::Vector)
  @timeit_debug ITensors.timer "bubble_sort" begin
    permutations = []
    n = length(v)
    for i in 1:n
      for j in 1:(n - i)
        if v[j] > v[j + 1]
          v[j], v[j + 1] = v[j + 1], v[j]
          push!(permutations, j)
        end
      end
    end
    return permutations
  end
end

function _bubble_sort(v1::Vector, v2::Vector)
  index_to_number = Dict()
  for (i, v) in enumerate(v2)
    index_to_number[v] = i
  end
  v1_num = [index_to_number[v] for v in v1]
  return _bubble_sort(v1_num)
end
