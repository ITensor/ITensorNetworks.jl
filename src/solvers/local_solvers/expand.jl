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
    #figure out next region, since we want to expand there
    #ToDo account for non-integer indices into which_region_update
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
    vp = region => next_vertex
  
    phi, has_changed = _two_site_expand_core(
      init, region, vp; projected_operator!, state!, expand_dir=1, updater_kwargs...
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
  expand_dir=1,
  svd_func_expand,
  cutoff,
  maxdim,
  use_relative_cutoff,
  use_absolute_cutoff,
)
  # preliminaries
  theflux = flux(phi0)
  svd_func = svd_func_expand
  v1 = first(vertexpair)
  v2 = last(vertexpair)
  verts = [v1, v2]
  n1, n2 = 1, 2
  PH = copy(projected_operator![])
  psi = copy(state![])
  # orthogonalize on the edge
  next_edge = edgetype(psi)(vertexpair)
  psi, phi = extract_local_tensor(psi, next_edge)
  psis = map(n -> psi[n], verts)  # extract local site tensors
  # ToDo: remove following code lines unless we want to truncate before we expand?
  ##this block is replaced by extract_local_tensor --- unless we want to truncate here this is the cleanest.
  ##otherwise reinsert the explicit block with truncation args (would allow to remove it from higher-level expander)
  #left_inds = uniqueinds(psis[n1], psis[n2])
  #U, S, V = svd(psis[n1], left_inds; lefttags=tags(commonind(psis[n1],psis[n2])), righttags=tags(commonind(psis[n1],psis[n2])))
  #psis[n1]= U
  #psi[region]=U
  #phi = S*V

  # don't expand if we are already at maxdim  
  old_linkdim = dim(commonind(first(psis), phi))
  linkinds = map(psi -> commonind(psi, phi), psis)
  (old_linkdim >= maxdim) && return phi0, false

  # compute nullspace to the left and right 
  @timeit_debug timer "nullvector" begin
    nullVecs = map(zip(psis, linkinds)) do (psi, linkind)
      #return nullspace(psi, linkind; atol=atol)
      return implicit_nullspace(psi, linkind)
    end
  end

  # update the projected operator 
  PH = set_nsite(PH, 2)
  PH = position(PH, psi, verts)

  # build environments
  g = underlying_graph(PH)
  @timeit_debug timer "build environments" begin
    envs = map(zip(verts, psis)) do (n, psi)
      return noprime(
        mapreduce(*, [v => n for v in neighbors(g, n) if !(v ∈ verts)]; init=psi) do e
          return PH.environments[NamedEdge(e)]
        end * PH.H[n],
      )
    end
  end

  # apply the projection into nullspace
  envs = [last(nullVecs)(last(envs)), first(nullVecs)(first(envs))]

  # assemble ITensorNetworkMap  #ToDo: do not rely on ITensorNetworkMap, simplify logic  
  ininds = uniqueinds(last(psis), phi)
  outinds = uniqueinds(first(psis), phi)
  cin = combiner(ininds)
  cout = combiner(outinds)
  envs = [cin * envs[1], cout * envs[2]]
  envMap=[last(envs), phi* first(envs)]
  #@show inds(envMap[1])
  #@show inds(envMap[2])
  
  
  
  #envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap(
  #  [last(envs), phi, first(envs)],
  #  uniqueinds(inds(cout), outinds),
  #  uniqueinds(inds(cin), ininds),
  #)
  #envMapDag = adjoint(envMap)

  # factorize
  @timeit_debug timer "svd_func" begin
    if svd_func == ITensorNetworks._svd_solve_normal
      U, S, V = svd_func(
        envMap, uniqueinds(inds(cout), outinds); maxdim=maxdim - old_linkdim, cutoff=cutoff
      )
    elseif svd_func == ITensorNetworks.rsvd_iterative
      
      U, S, V = svd_func(
        envMap,
        uniqueinds(inds(cout), outinds);
        maxdim=maxdim - old_linkdim,
        cutoff=cutoff,
        use_relative_cutoff=false,
        use_absolute_cutoff=true,
      )
      
      #U,S,V = svd_func(contract(envMap),uniqueinds(inds(cout),outinds);maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      #use_absolute_cutoff=true) #this one is for debugging in case we want to test the precontracted version
    else
      U, S, V = svd_func(
        eltype(envMap),
        envMap,
        envMapDag,
        uniqueinds(inds(cout), outinds);
        flux=theflux,
        maxdim=maxdim - old_linkdim,
        cutoff=cutoff,
      )
    end
  end

  # catch cases when we decompose a map of norm==0.0
  (isnothing(U) || iszero(norm(U))) && return phi0, false
  #FIXME: somehow the svd funcs sometimes return empty ITensors instead of nothing, that should be caught in the SVD routines instead...
  all(isempty.([U, S, V])) && return phi0, false #ToDo: do not rely on isempty here

  # uncombine indices on the non-link-indices
  U *= dag(cout)
  V *= dag(cin)
  #@show inds(U)
  #@show inds(V)
  #@show inds(S)
  
  # direct sum the site tensors
  @timeit_debug timer "direct sum" begin
    new_psis = map(zip(psis, [U, V])) do (psi, exp_basis)
      return ITensors.directsum(
        psi => commonind(psi, phi),
        exp_basis => uniqueind(exp_basis, psi);
        tags=tags(commonind(psi, phi)),
      )
    end
  end
  new_inds = [last(x) for x in new_psis]
  new_psis = [first(x) for x in new_psis]

  # extract the expanded linkinds from expanded site tensors and create a zero-tensor
  phi_indices = replace(
    inds(phi), (commonind(phi, psis[n]) => dag(new_inds[n]) for n in 1:2)...
  )
  if hasqns(phi)
    new_phi = ITensor(eltype(phi), flux(phi), phi_indices...)
    fill!(new_phi, 0.0)  #ToDo: Check whether this is really necessary.
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
    psi[v] = noprime(new_psi * C)
  end
  new_phi = dag(first(combiners)) * new_phi * dag(last(combiners))

  # apply expanded bond tensor to site tensor and reset projected operator to region
  psi[v1] *= new_phi
  new_phi = psi[v1]
  PH = set_nsite(PH, 1)
  PH = position(PH, psi, [region])

  projected_operator![] = PH
  state![] = psi
  return new_phi, true
  ##body end
end