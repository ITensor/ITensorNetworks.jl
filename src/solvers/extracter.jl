import ConstructionBase: setproperties

function extracter(problem, region_iterator; sweep, trunc=(;), kws...)
  trunc = truncation_parameters(sweep; trunc...)
  region = current_region(region_iterator)
  psi = itn.orthogonalize(state(problem), region)
  local_state = prod(psi[v] for v in region)
  problem = setproperties(problem; state=psi)

  problem, local_state = subspace_expand(
    problem, local_state, region_iterator; sweep, trunc, kws...
  )

  shifted_operator = itn.position(operator(problem), state(problem), region)

  return setproperties(problem; operator=shifted_operator), local_state
end
