using NDTensors: NDTensors
using NDTensors.BackendSelection: Backend, @Backend_str

@define_default_kwargs function subspace_expand!(
  region_iter, local_state; subspace_algorithm="nothing"
)
  backend = Backend(subspace_algorithm)

  if backend isa Backend"nothing"
    return local_state
  end

  subspace_expand!_kwargs = default_kwargs(
    subspace_expand!,
    backend,
    region_iter,
    local_state;
    region_kwargs(subspace_expand!, region_iter)...,
  )

  local_state = subspace_expand!(
    backend, region_iter, local_state; subspace_expand!_kwargs...
  )

  return local_state
end

function subspace_expand!(backend, region_iterator, local_state; kwargs...)
  # We allow passing of any kwargs here is this method throws an error anyway
  return error(
    "Subspace expansion (subspace_expand!) not defined for requested combination of subspace_algorithm and problem types",
  )
end

function compute_expansion(
  current_dim, basis_size; expansion_factor=1.5, maxexpand=typemax(Int), maxdim=typemax(Int)
)
  # Note: expand_maxdim will be *added* to current bond dimension
  # Obtain expand_maxdim from expansion_factor
  expand_maxdim = ceil(Int, expansion_factor * current_dim)
  # Enforce max_expand keyword
  expand_maxdim = min(maxexpand, expand_maxdim)

  # Restrict expand_maxdim below theoretical upper limit
  expand_maxdim = min(basis_size - current_dim, expand_maxdim)
  # Enforce total maxdim setting (e.g. used in insert step)
  expand_maxdim = min(maxdim - current_dim, expand_maxdim)
  # Ensure expand_maxdim is non-negative
  expand_maxdim = max(0, expand_maxdim)

  return expand_maxdim
end
