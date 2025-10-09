using NamedGraphs: edgetype

function insert!(region_iter, local_tensor; normalize=false, set_orthogonal_region=true)
  prob = problem(region_iter)

  region = current_region(region_iter)
  psi = copy(state(prob))
  if length(region) == 1
    C = local_tensor
  elseif length(region) == 2
    e = edgetype(psi)(first(region), last(region))
    indsTe = inds(psi[first(region)])
    tags = ITensors.tags(psi, e)

    U, C, spectrum = factorize(
      local_tensor, indsTe; tags, region_kwargs(factorize, region_iter)...
    )

    @preserve_graph psi[first(region)] = U
    prob = set_truncation_info!(prob; spectrum)
  else
    error("Region of length $(length(region)) not currently supported")
  end
  v = last(region)
  @preserve_graph psi[v] = C
  psi = set_orthogonal_region ? set_ortho_region(psi, [v]) : psi
  normalize && @preserve_graph psi[v] = psi[v] / norm(psi[v])

  prob.state = psi

  return prob
end
