@eval module $(gensym())
using Compat: Compat
using Graphs: vertices
using ITensorNetworks:
  ITensorNetworks,
  BeliefPropagationCache,
  IndsNetwork,
  ITensorNetwork,
  ⊗,
  apply,
  combine_linkinds,
  contract,
  contract_boundary_mps,
  contraction_sequence,
  eachtensor,
  environment,
  flatten_networks,
  linkinds_combiners,
  random_tensornetwork,
  siteinds,
  split_index,
  tensornetwork,
  update,
  update_factor
using ITensors: ITensors, ITensor, combiner, dag, inds, inner, op, prime, randomITensor
using ITensorNetworks.ModelNetworks: ModelNetworks
using ITensors.NDTensors: array
using LinearAlgebra: eigvals, tr
using NamedGraphs: NamedEdge
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NamedGraphs.PartitionedGraphs: PartitionVertex
using Random: Random
using SplitApplyCombine: group
using Test: @test, @testset

@testset "belief_propagation" begin
  ITensors.disable_warn_order()

  #First test on an MPS, should be exact
  g_dims = (1, 6)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  χ = 4
  Random.seed!(1234)
  ψ = random_tensornetwork(s; link_space=χ)

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  v = (1, 3)

  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  exact_sz = inner(Oψ, ψ) / inner(ψ, ψ)

  bpc = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
  bpc = update(bpc)
  env_tensors = environment(bpc, [PartitionVertex(v)])
  numerator = contract(vcat(env_tensors, ITensor[ψ[v], op("Sz", s[v]), dag(prime(ψ[v]))]))[]
  denominator = contract(vcat(env_tensors, ITensor[ψ[v], op("I", s[v]), dag(prime(ψ[v]))]))[]

  @test abs.((numerator / denominator) - exact_sz) <= 1e-14

  #Test updating the underlying tensornetwork in the cache
  v = first(vertices(ψψ))
  new_tensor = randomITensor(inds(ψψ[v]))
  bpc = update_factor(bpc, v, new_tensor)
  ψψ_updated = tensornetwork(bpc)
  @test ψψ_updated[v] == new_tensor

  #Now test on a tree, should also be exact
  g = named_comb_tree((4, 4))
  s = siteinds("S=1/2", g)
  χ = 2
  Random.seed!(1564)
  ψ = random_tensornetwork(s; link_space=χ)

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  v = (1, 3)

  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  exact_sz = inner(Oψ, ψ) / inner(ψ, ψ)

  bpc = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
  bpc = update(bpc)
  env_tensors = environment(bpc, [PartitionVertex(v)])
  numerator = contract(vcat(env_tensors, ITensor[ψ[v], op("Sz", s[v]), dag(prime(ψ[v]))]))[]
  denominator = contract(vcat(env_tensors, ITensor[ψ[v], op("I", s[v]), dag(prime(ψ[v]))]))[]

  @test abs.((numerator / denominator) - exact_sz) <= 1e-14

  #Now test two-site expec taking on the partition function of the Ising model. Not exact, but close
  g_dims = (3, 4)
  g = named_grid(g_dims)
  s = IndsNetwork(g; link_space=2)
  beta, h = 0.3, 0.5
  vs = [(2, 3), (3, 3)]
  ψψ = ModelNetworks.ising_network(s, beta; h)
  ψOψ = ModelNetworks.ising_network(s, beta; h, szverts=vs)

  contract_seq = contraction_sequence(ψψ)
  actual_szsz =
    contract(ψOψ; sequence=contract_seq)[] / contract(ψψ; sequence=contract_seq)[]

  bpc = BeliefPropagationCache(ψψ, group(v -> v, vertices(ψψ)))
  bpc = update(bpc; maxiter=20, verbose=true, tol=1e-5)

  env_tensors = environment(bpc, vs)
  numerator = contract(vcat(env_tensors, ITensor[ψOψ[v] for v in vs]))[]
  denominator = contract(vcat(env_tensors, ITensor[ψψ[v] for v in vs]))[]

  @test abs.((numerator / denominator) - actual_szsz) <= 0.05

  #Test forming a two-site RDM. Check it has the correct size, trace 1 and is PSD
  g_dims = (3, 3)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  vs = [(2, 2), (2, 3)]
  χ = 3
  ψ = random_tensornetwork(s; link_space=χ)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  bpc = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
  bpc = update(bpc; maxiter=20)

  ψψsplit = split_index(ψψ, NamedEdge.([(v, 1) => (v, 2) for v in vs]))
  env_tensors = environment(bpc, [(v, 2) for v in vs])
  rdm = contract(vcat(env_tensors, ITensor[ψψsplit[vp] for vp in [(v, 2) for v in vs]]))

  rdm = array((rdm * combiner(inds(rdm; plev=0)...)) * combiner(inds(rdm; plev=1)...))
  rdm /= tr(rdm)

  eigs = eigvals(rdm)
  @test size(rdm) == (2^length(vs), 2^length(vs))
  @test all(>=(0), real(eigs)) && all(==(0), imag(eigs))

  #Test more advanced block BP with MPS message tensors on a grid 
  g_dims = (4, 3)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  χ = 2
  ψ = random_tensornetwork(s; link_space=χ)
  v = (2, 2)

  ψψ = flatten_networks(ψ, dag(ψ); combine_linkinds=false, map_bra_linkinds=prime)
  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  ψOψ = flatten_networks(ψ, dag(Oψ); combine_linkinds=false, map_bra_linkinds=prime)

  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)
  ψOψ = combine_linkinds(ψOψ, combiners)

  bpc = BeliefPropagationCache(ψψ, group(v -> v[1], vertices(ψψ)))
  message_update_func(tns; kwargs...) = collect(
    eachtensor(first(contract(ITensorNetwork(tns); alg="density_matrix", kwargs...)))
  )
  bpc = update(
    bpc; message_update=message_update_func, message_update_kwargs=(; cutoff=1e-6, maxdim=4)
  )

  env_tensors = environment(bpc, [v])
  numerator = contract(vcat(env_tensors, ITensor[ψOψ[v]]))[]
  denominator = contract(vcat(env_tensors, ITensor[ψψ[v]]))[]

  exact_sz =
    contract_boundary_mps(ψOψ; cutoff=1e-16) / contract_boundary_mps(ψψ; cutoff=1e-16)

  @test abs.((numerator / denominator) - exact_sz) <= 1e-5
end
end
