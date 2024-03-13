# Here extract_local_tensor and insert_local_tensor
# are essentially inverse operations, adapted for different kinds of 
# algorithms and networks.
#

# sort of 2-site replacebond!; TODO: use dense TTN constructor instead
# ToDo: remove slurping of kwargs, fix kwargs
function insert_local_tensor(
  state::AbstractTTN,
  phi::ITensor,
  region,
  ortho_vert;
  normalize=false,
  # factorize kwargs
  maxdim=nothing,
  mindim=nothing,
  cutoff=nothing,
  which_decomp=nothing,
  eigen_perturbation=nothing,
  ortho=nothing,
  internal_kwargs,
  kwargs...,
)
  spec = nothing
  other_vertex = setdiff(support(region), [ortho_vert])
  if !isempty(other_vertex)
    v = only(other_vertex)
    e = edgetype(state)(ortho_vert, v)
    indsTe = inds(state[ortho_vert])
    L, phi, spec = factorize(
      phi,
      indsTe;
      tags=tags(state, e),
      maxdim,
      mindim,
      cutoff,
      which_decomp,
      eigen_perturbation,
      ortho,
    )
    state[ortho_vert] = L

  else
    v = ortho_vert
  end
  state[v] = phi
  state = set_ortho_center(state, [v])
  @assert isortho(state) && only(ortho_center(state)) == v
  normalize && (state[v] ./= norm(state[v]))
  return state, spec
end

# ToDo: remove slurping of kwargs, fix kwargs
function insert_local_tensor(
  state::AbstractTTN, phi::ITensor, region::NamedEdge, ortho; internal_kwargs, kwargs...
)
  v = only(setdiff(support(region), [ortho]))
  state[v] *= phi
  state = set_ortho_center(state, [v])
  return state, nothing
end
