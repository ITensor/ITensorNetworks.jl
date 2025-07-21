using ITensors:
  commonind,
  dag,
  dim,
  directsum,
  dot,
  hascommoninds,
  Index,
  norm,
  onehot,
  uniqueinds,
  random_itensor

# TODO: hoist num_expand default value out to a function or similar
function subspace_expand!(
  problem::EigsolveProblem, local_tensor, region; prev_region, num_expand=4, kws...
)
  if isnothing(prev_region) || isa(region, AbstractEdge)
    return local_tensor
  end

  prev_vertex_set = setdiff(prev_region, region)
  (length(prev_vertex_set) != 1) && return local_tensor
  prev_vertex = only(prev_vertex_set)

  psi = state(problem)
  A = psi[prev_vertex]

  next_vertex = only(filter(v -> (it.hascommoninds(psi[v], A)), region))
  C = psi[next_vertex]

  # Analyze indices of A
  # TODO: if "a" is missing, could supply a 1-dim index and put on both A and C?
  a = commonind(A, C)
  isnothing(a) && return local_tensor
  basis_inds = uniqueinds(A, C)

  # Determine maximum value of num_expand
  dim_basis = prod(dim.(basis_inds))
  num_expand = min(num_expand, dim_basis - dim(a))
  (num_expand <= 0) && return local_tensor

  # Build new subspace
  function linear_map(w)
    return w = w - A * (dag(A) * w)
  end
  random_vector() = random_itensor(basis_inds...)
  Q = range_finder(linear_map, random_vector; max_rank=num_expand, oversample=0)

  # Direct sum new space with A to make Ax
  qinds = [Index(1, "q$j") for j in 1:num_expand]
  Q = [Q[j] * onehot(qinds[j] => 1) => qinds[j] for j in 1:num_expand]
  Ax, sa = directsum(A => a, Q...)

  expander = dag(Ax) * A
  psi[prev_vertex] = Ax
  psi[next_vertex] = expander * C

  # TODO: avoid computing local tensor twice
  #       while also handling AbstractEdge region case
  local_tensor = prod(psi[v] for v in region)

  return local_tensor
end
