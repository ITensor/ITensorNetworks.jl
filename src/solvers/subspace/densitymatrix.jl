using NamedGraphs.GraphsExtensions: incident_edges
using Printf: @printf

@default_kwargs function subspace_expand!(
  ::Backend"densitymatrix", region_iter, local_state; north_pass=1
)
  prob = problem(region_iter)

  region = current_region(region_iter)
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
    dim(a), basis_size; region_kwargs(compute_expansion, region_iter)...
  )
  expanded_maxdim <= 0 && return local_state

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
  D, U = eigen(rho; region_kwargs(eigen, region_iter)..., ishermitian=true)

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
