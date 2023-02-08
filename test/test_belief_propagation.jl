using ITensorNetworks
using ITensorNetworks:
  ising_network,
  compute_message_tensors,
  nested_graph_leaf_vertices,
  calculate_contraction,
  split_index,
  contract_inner
using Test
using Compat
using ITensors
#ASSUME ONE CAN INSTALL THIS, MIGHT FAIL FOR WINDOWS
using Metis
using LinearAlgebra
using NamedGraphs
using SplitApplyCombine

@testset "belief_propagation" begin

  #FIRST TEST SINGLE SITE ON AN MPS, SHOULD BE ALMOST EXACT
  dims = (1, 6)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 4
  ψ = randomITensorNetwork(s; link_space=χ)

  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  v = (1, 3)

  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  exact_sz = contract_inner(Oψ, ψ) / contract_inner(ψ, ψ)

  nsites = 1
  vertex_groups = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=nsites)
  )
  mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)
  bp_sz =
    calculate_contraction(
      ψψ, mts, [(v, 1)]; verts_tensors=ITensor[apply(op("Sz", s[v]), ψ[v])]
    )[] / calculate_contraction(ψψ, mts, [(v, 1)])[]

  @test abs.(bp_sz - exact_sz) <= 0.00001

  #NOW TEST TWO_SITE_EXPEC TAKING ON THE PARTITION FUNCTION OF THE RECTANGULAR ISING. SHOULD BE REASONABLE AND 
  #INDEPENDENT OF INIT CONDITIONS, FOR SMALL BETA
  dims = (3, 4)
  g = named_grid(dims)
  s = IndsNetwork(g; link_space=2)
  beta = 0.2
  vs = [(2, 3), (3, 3)]
  ψψ = ising_network(s, beta)
  ψOψ = ising_network(s, beta; szverts=vs)

  contract_seq = contraction_sequence(ψψ)
  actual_szsz =
    ITensors.contract(ψOψ; sequence=contract_seq)[] /
    ITensors.contract(ψψ; sequence=contract_seq)[]

  nsites = 2
  mts = compute_message_tensors(ψψ; nvertices_per_partition=nsites)
  bp_szsz =
    calculate_contraction(ψψ, mts, vs; verts_tensors=[ψOψ[v] for v in vs])[] / calculate_contraction(ψψ, mts, vs)[]

  @test abs.(bp_szsz - actual_szsz) <= 0.05

  #FINALLY, TEST FORMING OF A TWO SITE RDM. JUST CHECK THAT IS HAS THE CORRECT SIZE, TRACE AND IS PSD
  dims = (4, 4)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  vs = [(2, 2), (2, 3), (2, 4)]
  χ = 3
  ψ = randomITensorNetwork(s; link_space=χ)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  nsites = 2
  vertex_groups = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=nsites)
  )
  mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)

  ψψsplit = split_index(ψψ, NamedEdge.([(v, 1) => (v, 2) for v in vs]))
  rdm = calculate_contraction(
    ψψ,
    mts,
    [(v, 2) for v in vs];
    verts_tensors=ITensor[ψψsplit[vp] for vp in [(v, 2) for v in vs]],
  )

  rdm = array((rdm * combiner(inds(rdm; plev=0)...)) * combiner(inds(rdm; plev=1)...))
  rdm /= tr(rdm)

  eigs = eigvals(rdm)
  @test all(>=(0), real(eigs)) && all(==(0), imag(eigs))
  @test size(rdm) == (2^length(vs), 2^length(vs))
end
