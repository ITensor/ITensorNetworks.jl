#ToDo: is this the correct scaling for infinite timestep? DMRG will not have infinite timestep, unless we set it explicitly
#ToDo: implement this as a closure, to be constructed in sweep_plan (where time_step is known)
#scale_cutoff_by_timestep(cutoff;time_step) = isinf(time_step) ? cutoff : cutoff / abs(time_step)

function two_site_expansion_updater(
    init;
    state!,
    projected_operator!,
    outputlevel,
    which_sweep,
    sweep_plan,
    which_region_update,
    internal_kwargs,
    svd_func_expand=ITensorNetworks.rsvd_iterative,
    maxdim,
    maxdim_func= (arg;kwargs...) -> identity(arg),
    cutoff,
    cutoff_func= (arg;kwargs...) -> identity(arg),
    use_relative_cutoff=false,
    use_absolute_cutoff=true,
  )
    maxdim=maxdim_func(maxdim;internal_kwargs...)
    cutoff=cutoff_func(cutoff;internal_kwargs...)
    region = first(sweep_plan[which_region_update])
    typeof(region) <: NamedEdge && return init, (;)
    region = only(region)
    # figure out next region, since we want to expand there
    # ToDo: account for non-integer indices into which_region_update
    next_region = if which_region_update == length(sweep_plan)
      nothing
    else
      first(sweep_plan[which_region_update + 1])
    end
    previous_region =
      which_region_update == 1 ? nothing : first(sweep_plan[which_region_update - 1])
    isnothing(next_region) && return init, (;)
    !(typeof(next_region) <: NamedEdge) && return init, (;)
    !(region == src(next_region) || region == dst(next_region)) && return init, (;)
    next_vertex = src(next_region) == region ? dst(next_region) : src(next_region)
  
    phi, has_changed = _two_site_expand_core(
      init, region, region => next_vertex; 
      projected_operator!,
      state!,
      svd_func_expand,
      maxdim,
      cutoff,
      use_relative_cutoff,
      use_absolute_cutoff,
    )
    !has_changed && return init, (;)
    return phi, (;)
  end

  """
Returns a function which applies Nullspace projector to an ITensor with matching indices without
explicitly constructing the projector as an ITensor or Network.
"""
function implicit_nullspace(A, linkind)
  # only works when applied in the direction of the environment tensor, not the other way (due to use of ITensorNetworkMap which is not reversible)
  outind = uniqueinds(A, linkind)
  inind = outind  #?
  # ToDo: implement without ITensorNetworkMap
  P = ITensors.ITensorNetworkMaps.ITensorNetworkMap(
    [prime(dag(A), linkind), prime(A, linkind)], inind, outind
  )
  return x::ITensor -> x - P(x)
end

"""
Performs a local expansion using the two-site projected variance.
The input state should have orthogonality center on a single vertex (region), and phi0 is that site_tensors.
Expansion is performed on vertex that would be visited next (if next vertex is a neighbor of region). 
"""
function _two_site_expand_core(
  phi0,
  region,
  vertexpair::Pair;
  projected_operator!,
  state!,
  svd_func_expand,
  cutoff,
  maxdim,
  use_relative_cutoff,
  use_absolute_cutoff,
)
  # preliminaries
  theflux = flux(phi0)
  v1 = first(vertexpair)
  v2 = last(vertexpair)
  verts = [v1, v2]
  PH = copy(projected_operator![])
  state = copy(state![])
  old_linkdim = dim(commonind(state[v1],state[v2]))
  (old_linkdim >= maxdim) && return phi0, false
  # orthogonalize on the edge
  #update of environment not strictly necessary here
  state, PH, phi = default_extracter(state, PH, edgetype(state)(vertexpair), v1;
                                  internal_kwargs=(;))
  psis = map(n -> state[n], verts)  # extract local site tensors
  linkinds = map(psi -> commonind(psi, phi), psis)

  # compute nullspace to the left and right 
  #@timeit_debug timer "nullvector" begin
    nullVecs = map(zip(psis, linkinds)) do (psi, linkind)
      #return nullspace(psi, linkind; atol=atol)
      return ITensorNetworks.implicit_nullspace(psi, linkind)
    end
  #end

  # update the projected operator
  #ToDo: may not be necessary if we use the extracter anyway
  PH = set_nsite(PH, 2)
  PH = position(PH, state, verts)

  # build environments
  g = underlying_graph(PH)
  #@timeit_debug timer "build environments" begin
    envs = map(zip(verts, psis)) do (n, psi)
      return noprime(
        mapreduce(*, [v => n for v in neighbors(g, n) if !(v ∈ verts)]; init=psi) do e
          return PH.environments[NamedEdge(e)]
        end * PH.operator[n],
      )
    end
  #end

  # apply the projection into nullspace,
  # FIXME: think through again whether we really want to evaluate null space projector already
  envs=[ first(nullVecs)(first(envs)),last(nullVecs)(last(envs))]
  ininds = uniqueinds(last(psis), phi)
  outinds = uniqueinds(first(psis), phi)
  cin = combiner(ininds)
  cout = combiner(outinds)
  envMap=[cout * envs[1], phi* (cin * envs[2])]

  # factorize
  #@timeit_debug timer "svd_func" begin
  U, S, V = svd_func_expand(
        envMap,
        uniqueinds(inds(cout), outinds);
        maxdim=maxdim - old_linkdim,
        cutoff=cutoff,
        use_relative_cutoff,
        use_absolute_cutoff,
        )
  #end

  # catch cases when we decompose a map of norm==0.0
  (isnothing(U) || iszero(norm(U))) && return phi0, false
  #FIXME: somehow the svd funcs sometimes return empty ITensors instead of nothing, that should be caught in the SVD routines instead...
  all(isempty.([U, S, V])) && return phi0, false #ToDo: do not rely on isempty here

  # uncombine indices on the non-link-indices
  U *= dag(cout)
  V *= dag(cin)

  # direct sum the site tensors
  #@timeit_debug timer "direct sum" begin
    new_psis = map(zip(psis, [U, V])) do (psi, exp_basis)
      return ITensors.directsum(
        psi => commonind(psi, phi),
        exp_basis => uniqueind(exp_basis, psi);
        tags=tags(commonind(psi, phi)),
      )
    end
  #end
  new_inds = [last(x) for x in new_psis]
  new_psis = [first(x) for x in new_psis]

  # extract the expanded linkinds from expanded site tensors and create a zero-tensor
  phi_indices = replace(
    inds(phi), (commonind(phi, psis[n]) => dag(new_inds[n]) for n in 1:2)...
  )
  if hasqns(phi)
    new_phi = ITensor(eltype(phi), flux(phi), phi_indices...)
    #ToDo: Remove? This used to be necessary, but may have been fixed with bugfixes in ITensor
    #fill!(new_phi, 0.0)
  else
    new_phi = ITensor(eltype(phi), phi_indices...)
  end

  # set the new bond tensor elements from the old bond tensor
  map(eachindex(phi)) do I
    v = phi[I]
    !iszero(v) && (return new_phi[I] = v) # I think this line errors without the fill! with zeros above
  end

  # apply combiners on linkinds #ToDo: figure out why this is strictly necessary 
  combiners = map(
    new_ind -> combiner(new_ind; tags=tags(new_ind), dir=dir(new_ind)), new_inds
  )
  for (v, new_psi, C) in zip(verts, new_psis, combiners)
    state[v] = noprime(new_psi * C)
  end
  new_phi = dag(first(combiners)) * new_phi * dag(last(combiners))

  # apply expanded bond tensor to site tensor and reset projected operator to region
  state[v1] *= new_phi
  new_phi = state[v1]
  PH = set_nsite(PH, 1)
  PH = position(PH, state, [region])
  projected_operator![] = PH
  state![] = state
  return new_phi, true
end
