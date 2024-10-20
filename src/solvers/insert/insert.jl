# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.

# TODO: use dense TTN constructor to make this more general.
function default_inserter(
  state::AbstractTTN,
  phi::ITensor,
  region;
  normalize=false,
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  internal_kwargs,
)
  state = copy(state)
  spec = nothing
  if length(region) == 2
    v = last(region)
    e = edgetype(state)(first(region), last(region))
    indsTe = inds(state[first(region)])
    L, phi, spec = factorize(phi, indsTe; tags=tags(state, e), maxdim, mindim, cutoff)
    state[first(region)] = L
  else
    v = only(region)
  end
  state[v] = phi
  state = set_ortho_region(state, [v])
  normalize && (state[v] /= norm(state[v]))
  return state, spec
end

function default_inserter(
  state::AbstractTTN,
  phi::ITensor,
  region::NamedEdge;
  cutoff=nothing,
  maxdim=nothing,
  mindim=nothing,
  normalize=false,
  internal_kwargs,
)
  state[dst(region)] *= phi
  state = set_ortho_region(state, [dst(region)])
  return state, nothing
end
