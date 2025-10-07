function extract!(iter; kwargs...)
  return _extract_fallback!(iter; subspace_algorithm="nothing", kwargs...)
end

# Internal function such that a method error can be thrown while still allowing a user
# to specialize on `extract!`
function _extract_fallback!(region_iter::RegionIterator; subspace_algorithm)
  prob = problem(region_iter)
  region = current_region(region_iter)

  psi = orthogonalize(state(prob), region)
  local_state = prod(psi[v] for v in region)

  prob.state = psi

  local_state = subspace_expand!(local_state, region_iter; subspace_algorithm)
  shifted_operator = position(operator(prob), state(prob), region)

  prob.operator = shifted_operator

  return local_state
end
