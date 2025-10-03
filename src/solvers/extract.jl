function extract!(region_iterator; sweep, trunc=(;), kws...)
  prob = problem(region_iterator)

  trunc = truncation_parameters(sweep; trunc...)
  region = current_region(region_iterator)
  psi = orthogonalize(state(prob), region)
  local_state = prod(psi[v] for v in region)

  prob.state = psi

  local_state = subspace_expand!(local_state, region_iterator; sweep, trunc, kws...)

  shifted_operator = position(operator(prob), state(prob), region)

  prob.operator = shifted_operator

  return local_state
end
