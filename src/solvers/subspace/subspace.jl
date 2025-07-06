using NDTensors: NDTensors
using NDTensors.BackendSelection: Backend, @Backend_str

default_expansion_factor() = 1.5
default_max_expand() = typemax(Int)

function subspace_expand(
  problem, local_state, region_iterator; subspace_algorithm=nothing, sweep, trunc, kws...
)
  return subspace_expand(
    Backend(subspace_algorithm), problem, local_state, region_iterator; trunc, kws...
  )
end

function subspace_expand(backend, problem, local_state, region_iterator; kws...)
  error(
    "Subspace expansion (subspace_expand!) not defined for requested combination of subspace_algorithm and problem types",
  )
end

function subspace_expand(
  backend::Backend{:nothing}, problem, local_state, region_iterator; kws...
)
  problem, local_state
end

function compute_expansion(
  current_dim,
  basis_size;
  expansion_factor=default_expansion_factor(),
  max_expand=default_max_expand(),
  maxdim=default_maxdim(),
)
  # Note: expand_maxdim will be *added* to current bond dimension
  # Obtain expand_maxdim from expansion_factor
  expand_maxdim = ceil(Int, expansion_factor * current_dim)
  # Enforce max_expand keyword
  expand_maxdim = min(max_expand, expand_maxdim)

  # Restrict expand_maxdim below theoretical upper limit
  expand_maxdim = min(basis_size-current_dim, expand_maxdim)
  # Enforce total maxdim setting (e.g. used in insert step)
  expand_maxdim = min(maxdim-current_dim, expand_maxdim)
  # Ensure expand_maxdim is non-negative
  expand_maxdim = max(0, expand_maxdim)
  return expand_maxdim
end
