import ConstructionBase: setproperties

function inserter(
  problem,
  local_tensor,
  region_iterator;
  normalize=false,
  set_orthogonal_region=true,
  sweep,
  trunc=(;),
  kws...,
)
  trunc = truncation_parameters(sweep; trunc...)

  region = current_region(region_iterator)
  psi = copy(state(problem))
  if length(region) == 1
    C = local_tensor
  elseif length(region) == 2
    e = ng.edgetype(psi)(first(region), last(region))
    indsTe = it.inds(psi[first(region)])
    tags = it.tags(psi, e)
    U, C, _ = it.factorize(local_tensor, indsTe; tags, trunc...)
    itn.@preserve_graph psi[first(region)] = U
  else
    error("Region of length $(length(region)) not currently supported")
  end
  v = last(region)
  itn.@preserve_graph psi[v] = C
  psi = set_orthogonal_region ? itn.set_ortho_region(psi, [v]) : psi
  normalize && itn.@preserve_graph psi[v] = psi[v] / norm(psi[v])
  return setproperties(problem; state=psi)
end
