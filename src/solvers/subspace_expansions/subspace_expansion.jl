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
