# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.

# TODO: use dense TTN constructor to make this more general.
function default_inserter(
  state::AbstractTTN,
  phi::ITensor,
  region,
  ortho_vert;
  normalize=false,
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  internal_kwargs,
)
  state = copy(state)
  spec = nothing
  other_vertex = setdiff(support(region), [ortho_vert])
  if !isempty(other_vertex)
    v = only(other_vertex)
    e = edgetype(state)(ortho_vert, v)
    indsTe = inds(state[ortho_vert])
    L, phi, spec = factorize(phi, indsTe; tags=tags(state, e), maxdim, mindim, cutoff)
    state[ortho_vert] = L

  else
    v = ortho_vert
  end
  state[v] = phi
  state = set_ortho_region(state, [v])
  normalize && (state[v] /= norm(state[v]))
  return state, spec
end

function default_inserter(
  state::AbstractTTN,
  phi::ITensor,
  region::NamedEdge,
  ortho;
  cutoff=nothing,
  maxdim=nothing,
  mindim=nothing,
  normalize=false,
  internal_kwargs,
)
  v = only(setdiff(support(region), [ortho]))
  state[v] *= phi
  state = set_ortho_region(state, [v])
  return state, nothing
end
