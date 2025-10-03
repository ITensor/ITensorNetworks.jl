using NamedGraphs.GraphsExtensions: incident_edges
using Printf: @printf

function subspace_expand!(
  ::Backend"densitymatrix",
  local_state::ITensor,
  region_iterator;
  expansion_factor,
  max_expand,
  north_pass=1,
  trunc,
  kws...,
)
  prob = problem(region_iterator)

  region = current_region(region_iterator)
  psi = copy(state(prob))

  prev_vertex_set = setdiff(pos(operator(prob)), region)
  (length(prev_vertex_set) != 1) && return local_state
  prev_vertex = only(prev_vertex_set)
  A = psi[prev_vertex]

  next_vertices = filter(v -> (hascommoninds(psi[v], A)), region)
  isempty(next_vertices) && return local_state
  next_vertex = only(next_vertices)
  C = psi[next_vertex]

  a = commonind(A, C)
  isnothing(a) && return local_state
  basis_size = prod(dim.(uniqueinds(A, C)))

  expanded_maxdim = compute_expansion(
    dim(a), basis_size; expansion_factor, max_expand, trunc.maxdim
  )
  expanded_maxdim <= 0 && return local_state
  trunc = (; trunc..., maxdim=expanded_maxdim)

  envs = environments(operator(prob))
  H = operator(operator(prob))
  sqrt_rho = A
  for e in incident_edges(operator(prob))
    (src(e) ∈ region || dst(e) ∈ region) && continue
    sqrt_rho *= envs[e]
  end
  sqrt_rho *= H[prev_vertex]

  conj_proj_A(T) = (T - prime(A) * (dag(prime(A)) * T))
  for pass in 1:north_pass
    sqrt_rho = conj_proj_A(sqrt_rho)
  end
  rho = sqrt_rho * dag(noprime(sqrt_rho))
  D, U = eigen(rho; trunc..., ishermitian=true)

  Uproj(T) = (T - prime(A, a) * (dag(prime(A, a)) * T))
  for pass in 1:north_pass
    U = Uproj(U)
  end
  if norm(dag(U) * A) > 1E-10
    @printf("Warning: |U*A| = %.3E in subspace expansion\n", norm(dag(U) * A))
    return local_state
  end

  Ax, ax = directsum(A => a, U => commonind(U, D))
  expander = dag(Ax) * A
  psi[prev_vertex] = Ax
  psi[next_vertex] = expander * C
  local_state = expander * local_state

  prob.state = psi

  return local_state
end
