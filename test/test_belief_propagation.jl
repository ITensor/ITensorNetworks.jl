using ITensorNetworks
using ITensorNetworks:
  ising_network,
  belief_propagation,
  split_index,
  contract_inner,
  contract_boundary_mps,
  message_tensors,
  environment_tensors
using Test
using Compat
using ITensors
using LinearAlgebra
using NamedGraphs
using SplitApplyCombine
using Random
using Metis

ITensors.disable_warn_order()

@testset "belief_propagation" begin

  #First test on an MPS, should be exact
  g_dims = (1, 6)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  χ = 4
  Random.seed!(1234)
  ψ = randomITensorNetwork(s; link_space=χ)

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  v = (1, 3)

  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  exact_sz = contract_inner(Oψ, ψ) / contract_inner(ψ, ψ)

  pψψ = PartitionedGraph(ψψ, group(v -> v[1], vertices(ψψ)))
  mts = belief_propagation(pψψ)
  env_tensors = environment_tensors(pψψ, mts, [PartitionVertex(v)])
  numerator = contract(vcat(env_tensors, ITensor[ψ[v], op("Sz", s[v]), dag(prime(ψ[v]))]))[]
  denominator = contract(vcat(env_tensors, ITensor[ψ[v], op("I", s[v]), dag(prime(ψ[v]))]))[]

  @test abs.((numerator / denominator) - exact_sz) <= 1e-14

  #Now test on a tree, should also be exact
  g = named_comb_tree((4, 4))
  s = siteinds("S=1/2", g)
  χ = 2
  Random.seed!(1564)
  ψ = randomITensorNetwork(s; link_space=χ)

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  v = (1, 3)

  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  exact_sz = contract_inner(Oψ, ψ) / contract_inner(ψ, ψ)

  pψψ = PartitionedGraph(ψψ, group(v -> v[1], vertices(ψψ)))
  mts = belief_propagation(pψψ)
  env_tensors = environment_tensors(pψψ, mts, [PartitionVertex(v)])
  numerator = contract(vcat(env_tensors, ITensor[ψ[v], op("Sz", s[v]), dag(prime(ψ[v]))]))[]
  denominator = contract(vcat(env_tensors, ITensor[ψ[v], op("I", s[v]), dag(prime(ψ[v]))]))[]

  @test abs.((numerator / denominator) - exact_sz) <= 1e-14

  #Now test two-site expec taking on the partition function of the Ising model. Not exact, but close
  g_dims = (3, 4)
  g = named_grid(g_dims)
  s = IndsNetwork(g; link_space=2)
  beta = 0.2
  vs = [(2, 3), (3, 3)]
  ψψ = ising_network(s, beta)
  ψOψ = ising_network(s, beta; szverts=vs)

  contract_seq = contraction_sequence(ψψ)
  actual_szsz =
    ITensors.contract(ψOψ; sequence=contract_seq)[] /
    ITensors.contract(ψψ; sequence=contract_seq)[]

  pψψ = PartitionedGraph(ψψ; nvertices_per_partition=2, backend="Metis")
  mts = belief_propagation(pψψ; niters=20)

  env_tensors = environment_tensors(pψψ, mts, vs)
  numerator = contract(vcat(env_tensors, ITensor[ψOψ[v] for v in vs]))[]
  denominator = contract(vcat(env_tensors, ITensor[ψψ[v] for v in vs]))[]

  @test abs.((numerator / denominator) - actual_szsz) <= 0.05

  #Test forming a two-site RDM. Check it has the correct size, trace 1 and is PSD
  g_dims = (3, 3)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  vs = [(2, 2), (2, 3)]
  χ = 3
  ψ = randomITensorNetwork(s; link_space=χ)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  pψψ = PartitionedGraph(ψψ, group(v -> v[1], vertices(ψψ)))
  mts = belief_propagation(pψψ; niters=20)

  ψψsplit = split_index(ψψ, NamedEdge.([(v, 1) => (v, 2) for v in vs]))
  env_tensors = environment_tensors(pψψ, mts, [(v, 2) for v in vs])
  rdm = ITensors.contract(
    vcat(env_tensors, ITensor[ψψsplit[vp] for vp in [(v, 2) for v in vs]])
  )

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
  ψ = randomITensorNetwork(s; link_space=χ)
  maxdim = 8
  v = (2, 2)

  ψψ = flatten_networks(ψ, dag(ψ); combine_linkinds=false, map_bra_linkinds=prime)
  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  ψOψ = flatten_networks(ψ, dag(Oψ); combine_linkinds=false, map_bra_linkinds=prime)

  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)
  ψOψ = combine_linkinds(ψOψ, combiners)
  pψψ = PartitionedGraph(ψψ, group(v -> v[1], vertices(ψψ)))
  mts = belief_propagation(
    pψψ;
    contract_kwargs=(;
      alg="density_matrix", output_structure=path_graph_structure, cutoff=1e-12, maxdim
    ),
  )

  env_tensors = environment_tensors(pψψ, mts, [v])
  numerator = contract(vcat(env_tensors, ITensor[ψOψ[v]]))[]
  denominator = contract(vcat(env_tensors, ITensor[ψψ[v]]))[]

  exact_sz =
    contract_boundary_mps(ψOψ; cutoff=1e-16) / contract_boundary_mps(ψψ; cutoff=1e-16)

  @test abs.((numerator / denominator) - exact_sz) <= 1e-5
end
