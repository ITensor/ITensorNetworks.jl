@eval module $(gensym())
using Compat: Compat
using Graphs: vertices
# Trigger package extension.
using ITensorNetworks:
  ITensorNetworks,
  BeliefPropagationCache,
  ⊗,
  combine_linkinds,
  contract,
  contract_boundary_mps,
  contraction_sequence,
  eachtensor,
  environment,
  inner_network,
  linkinds_combiners,
  message,
  partitioned_tensornetwork,
  random_tensornetwork,
  scalar,
  siteinds,
  split_index,
  tensornetwork,
  update,
  update_factor,
  update_message
using ITensors: ITensors, ITensor, combiner, dag, inds, inner, op, prime, random_itensor
using ITensorNetworks.ModelNetworks: ModelNetworks
using ITensors.NDTensors: array
using LinearAlgebra: eigvals, tr
using NamedGraphs: NamedEdge, NamedGraph
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NamedGraphs.PartitionedGraphs: PartitionVertex, partitionedges
using SplitApplyCombine: group
using StableRNGs: StableRNG
using Test: @test, @testset
@testset "belief_propagation" begin
  ITensors.disable_warn_order()
  g = named_grid((3, 3))
  s = siteinds("S=1/2", g)
  χ = 2
  rng = StableRNG(1234)
  ψ = random_tensornetwork(rng, s; link_space=χ)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])
  bpc = BeliefPropagationCache(ψψ)
  bpc = update(bpc; maxiter=50, tol=1e-10)
  #Test messages are converged
  for pe in partitionedges(partitioned_tensornetwork(bpc))
    @test update_message(bpc, pe) ≈ message(bpc, pe) atol = 1e-8
  end
  #Test updating the underlying tensornetwork in the cache
  v = first(vertices(ψψ))
  rng = StableRNG(1234)
  new_tensor = random_itensor(rng, inds(ψψ[v]))
  bpc_updated = update_factor(bpc, v, new_tensor)
  ψψ_updated = tensornetwork(bpc_updated)
  @test ψψ_updated[v] == new_tensor

  #Test forming a two-site RDM. Check it has the correct size, trace 1 and is PSD
  vs = [(2, 2), (2, 3)]

  ψψsplit = split_index(ψψ, NamedEdge.([(v, 1) => (v, 2) for v in vs]))
  env_tensors = environment(bpc, [(v, 2) for v in vs])
  rdm = contract(vcat(env_tensors, ITensor[ψψsplit[vp] for vp in [(v, 2) for v in vs]]))

  rdm = array((rdm * combiner(inds(rdm; plev=0)...)) * combiner(inds(rdm; plev=1)...))
  rdm /= tr(rdm)

  eigs = eigvals(rdm)
  @test size(rdm) == (2^length(vs), 2^length(vs))

  @test all(eig -> imag(eig) ≈ 0, eigs)
  @test all(eig -> real(eig) >= -eps(eltype(eig)), eigs)

  #Test edge case of network which evalutes to 0
  χ = 2
  g = named_grid((3, 1))
  rng = StableRNG(1234)
  ψ = random_tensornetwork(rng, ComplexF64, g; link_space=χ)
  ψ[(1, 1)] = 0.0 * ψ[(1, 1)]
  @test iszero(scalar(ψ; alg="bp"))
end
end
