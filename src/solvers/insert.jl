using NamedGraphs: edgetype

function insert(
  problem,
  local_tensor,
  region_iterator;
  normalize=false,
  set_orthogonal_region=true,
  sweep,
  trunc=(;),
  outputlevel=0,
  kws...,
)
  trunc = truncation_parameters(sweep; trunc...)
  region = current_region(region_iterator)
  psi = copy(state(problem))
  if length(region) == 1
    C = local_tensor
  elseif length(region) == 2
    e = edgetype(psi)(first(region), last(region))
    indsTe = inds(psi[first(region)])
    tags = ITensors.tags(psi, e)
    U, C, spectrum = factorize(local_tensor, indsTe; tags, trunc...)
    @preserve_graph psi[first(region)] = U
    problem = set_truncation_info(problem; spectrum)
  else
    error("Region of length $(length(region)) not currently supported")
  end
  v = last(region)
  @preserve_graph psi[v] = C
  psi = set_orthogonal_region ? set_ortho_region(psi, [v]) : psi
  normalize && @preserve_graph psi[v] = psi[v] / norm(psi[v])
  problem = set_state(problem, psi)
  return problem
end
