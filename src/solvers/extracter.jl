function extracter(problem, region_iterator; sweep, trunc=(;), kws...)
  trunc = truncation_parameters(sweep; trunc...)
  region = current_region(region_iterator)
  psi = orthogonalize(state(problem), region)
  local_state = prod(psi[v] for v in region)
  problem = set_state(problem, psi)
  problem, local_state = subspace_expand(
    problem, local_state, region_iterator; sweep, trunc, kws...
  )
  shifted_operator = position(operator(problem), state(problem), region)
  return set_operator(problem, shifted_operator), local_state
end
