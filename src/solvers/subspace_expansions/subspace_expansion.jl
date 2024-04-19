function two_site_expansion_updater(
  init;
  state!,
  projected_operator!,
  outputlevel,
  which_sweep,
  sweep_plan,
  which_region_update,
  region_kwargs,
  updater_kwargs,
)
  (; maxdim, cutoff, time_step) = region_kwargs
  #ToDo: handle timestep==Inf for DMRG case
  default_updater_kwargs = (;
    svd_func_expand=rsvd_iterative,
    maxdim=(maxdim == typemax(Int) ? maxdim : Int(ceil((2.0^(1 / 3))) * maxdim)),
    cutoff=isinf(time_step) ? cutoff : cutoff / abs(time_step), # ToDo verify that this is the correct way of scaling the cutoff
    use_relative_cutoff=false,
    use_absolute_cutoff=true,
  )
  updater_kwargs = merge(default_updater_kwargs, updater_kwargs)

  # if on edge return without doing anything
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
  #@timeit_debug timer "nullvector" begin
  nullVecs = map(zip(psis, linkinds)) do (psi, linkind)
    #return nullspace(psi, linkind; atol=atol)
    return implicit_nullspace(psi, linkind)
  end
  #end

  # update the projected operator 
  PH = set_nsite(PH, 2)
  PH = position(PH, psi, verts)

  # build environments
  g = underlying_graph(PH)
  #@timeit_debug timer "build environments" begin
  envs = map(zip(verts, psis)) do (n, psi)
    return noprime(
      mapreduce(*, [v => n for v in neighbors(g, n) if !(v ∈ verts)]; init=psi) do e
        return PH.environments[NamedEdge(e)]
      end * PH.H[n],
    )
  end
  #end

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
  #@timeit_debug timer "svd_func" begin
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
  #end

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

#=
function _full_expand_core_vertex(
    PH, psi, phi, region, svd_func; direction, expand_dir=-1, expander_cache=Any[], maxdim, cutoff, atol=1e-8, kwargs...,
) ###ToDo: adapt to current interface, broken as of now.
  #@show cutoff
  #enforce expand_dir in the tested direction
  @assert expand_dir==-1
  #println("in full expand")
  ### only on edges
  #(typeof(region)!=NamedEdge{Int}) && return psi, phi, PH
  ### only on verts
  (typeof(region)==NamedEdge{Int}) && return psi, phi, PH
  ## kind of hacky - only works for mps. More general?
  n1 = first(region)
  theflux=flux(psi[n1])
  #expand_dir=-1
  if direction == 1 
    n2 = expand_dir == 1 ? n1 - 1 : n1+1
  else
    n2 = expand_dir == 1 ? n1 + 1 : n1-1
  end

  (n2 < 1 || n2 > length(vertices(psi))) && return psi,phi,PH
  neutralflux=flux(psi[n2])

  verts = [n1,n2]

  if isempty(expander_cache)
    @warn("building environment of H^2 from scratch!")

    g = underlying_graph(PH.H)
    H = vertex_data(data_graph(PH.H))

    H_dag = swapprime.(prime.(dag.(H)), 1,2, tags = "Site")
    H2_vd= replaceprime.(map(*, H, H_dag), 2 => 1)
    H2_ed = edge_data(data_graph(PH.H))

    H2 = TTN(ITensorNetwork(DataGraph(g, H2_vd, H2_ed)), PH.H.ortho_center)
    PH2 = ProjTTN(H2)
    PH2 = set_nsite(PH2, 2)

    push!(expander_cache, PH2)
  end

  PH2 = expander_cache[1]
  n1 = verts[2]
  n2 = verts[1]

  ###do we really need to make the environments two-site?
  ###look into how ProjTTN works
  PH2 = position(PH2, psi, [n1,n2])
  PH = position(PH, psi, [n1,n2])

  psi1 = psi[n1]  
  psi2 = psi[n2] #phi
  old_linkdim = dim(commonind(psi1, psi2))

  # don't expand if we are already at maxdim
  ## make more transparent that this is not the normal maxdim arg but maxdim_expand
  ## when would this even happen?
  #@show old_linkdim, maxdim

  old_linkdim >= maxdim && return psi, phi, PH
  #@show "expandin", maxdim, old_linkdim, maxdim-old_linkdim
  # compute nullspace to the left and right 
  linkind_l = commonind(psi1, psi2)
  nullVec = implicit_nullspace(psi1, linkind_l)

  # if nullspace is empty (happen's for product states with QNs)
  ## ToDo implement norm or equivalent for a generic LinearMap (i guess via sampling a random vector)
  #norm(nullVec) == 0.0 && return psi, phi, PH

  ## compute both environments
  g = underlying_graph(PH)

  @timeit_debug timer "build environments" begin
    env1 = noprime(mapreduce(*, [v => n1 for v in neighbors(g, n1) if v != n2]; init = psi1) do e
      return PH.environments[NamedEdge(e)]
    end *PH.H[n1]
    )
    env2p = noprime(mapreduce(*, [v => n2 for v in neighbors(g, n2) if v != n1]; init = psi2) do e
      return PH.environments[NamedEdge(e)]
    end *PH.H[n2]
    )

    env2 = mapreduce(*, [v => n2 for v in neighbors(g, n2) if v != n1]; init = psi2) do e
      return PH2.environments[NamedEdge(e)]
    end * PH2.H[n2]*prime(dag(psi2))
  end

  env1=nullVec(env1)

  outinds = uniqueinds(psi1,psi2)
  ininds = dag.(outinds)
  cout=combiner(outinds)
  env1 *= cout
  env2p *= prime(dag(psi[n2]),commonind(dag(psi[n2]),dag(psi[n1])))
  env2p2= replaceprime(env2p * replaceprime(dag(env2p),0=>2),2=>1)

  envMap = ITensors.ITensorNetworkMaps.ITensorNetworkMap([prime(dag(env1)),(env2-env2p2),env1] , prime(dag(uniqueinds(cout,outinds))), uniqueinds(cout,outinds))
  envMapDag=adjoint(envMap)
  # svd-decomposition
  @timeit_debug timer "svd_func" begin
    if svd_func==ITensorNetworks._svd_solve_normal
      U,S,_ = svd_func(envMap, uniqueinds(inds(cout),outinds); maxdim=maxdim-old_linkdim, cutoff=cutoff)
    elseif svd_func==ITensorNetworks.rsvd_iterative
      #@show theflux
      envMap=transpose(envMap)
      U,S,_ = svd_func(eltype(first(envMap.itensors)),envMap,ITensors.ITensorNetworkMaps.input_inds(envMap);theflux=neutralflux, maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      use_absolute_cutoff=true)
      #U,S,V =  svd_func(contract(envMap),uniqueinds(inds(cout),outinds);theflux=theflux, maxdim=maxdim-old_linkdim, cutoff=cutoff, use_relative_cutoff=false,
      #use_absolute_cutoff=true)
    else
      U,S,_= svd_func(eltype(envMap),envMap,envMapDag,uniqueinds(cout,outinds); flux=neutralflux, maxdim=maxdim-old_linkdim, cutoff=cutoff)
    end
  end
  isnothing(U) && return psi,phi,PH
  ###FIXME: somehow the svd funcs sometimes return empty ITensors instead of nothing, that should be caught in the SVD routines instead...
  all(isempty.([U,S])) && return psi, phi, PH
  @assert dim(commonind(U, S)) ≤ maxdim-old_linkdim
  nullVec = dag(cout)*U
  new_psi1, new_ind1 = ITensors.directsum(
    psi1 => uniqueinds(psi1, nullVec), nullVec => uniqueinds(nullVec, psi1); tags=(tags(commonind(psi1,phi)),)
  )
  new_ind1 = new_ind1[1]
  @assert dim(new_ind1) <= maxdim

  Cl = combiner(new_ind1; tags=tags(new_ind1), dir=dir(new_ind1))

  phi_indices = replace(inds(phi), commonind(phi,psi1) => dag(new_ind1))

  if hasqns(phi)
    new_phi=ITensor(eltype(phi),flux(phi), phi_indices...)
    fill!(new_phi,0.0)
  else
    new_phi = ITensor(eltype(phi), phi_indices...)
  end

  map(eachindex(phi)) do I
    v = phi[I]
    !iszero(v) && (return new_phi[I]=v)
  end

  psi[n1] = noprime(new_psi1*Cl)
  new_phi = dag(Cl)*new_phi

  return psi, new_phi, PH
end
=#
