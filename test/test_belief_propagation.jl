using ITensorNetworks
using ITensorNetworks:
  ising_network,
  compute_message_tensors,
  nested_graph_leaf_vertices,
  calculate_contraction_network,
  split_index,
  contract_inner,
  contract_boundary_mps
using Test
using Compat
using ITensors
#ASSUME ONE CAN INSTALL THIS, MIGHT FAIL FOR WINDOWS
using Metis
using LinearAlgebra
using NamedGraphs
using SplitApplyCombine
using Random

ITensors.disable_warn_order()

@testset "belief_propagation" begin

  #FIRST TEST SINGLE SITE ON AN MPS, SHOULD BE EXACT
  dims = (1, 6)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 4
  Random.seed!(1234)
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
  numerator_network = calculate_contraction_network(
    ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork(ITensor[apply(op("Sz", s[v]), ψ[v])])
  )
  denominator_network = calculate_contraction_network(ψψ, mts, [(v, 1)])
  bp_sz = contract(numerator_network)[]/contract(denominator_network)[]

  @test abs.(bp_sz - exact_sz) <= 1e-14

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
  numerator_network = calculate_contraction_network(
    ψψ, mts, vs; verts_tn=ITensorNetwork(ITensor[ψOψ[v] for v in vs])
  )
  denominator_network = calculate_contraction_network(ψψ, mts, vs)
  bp_szsz = contract(numerator_network)[]/contract(denominator_network)[]

  @test abs.(bp_szsz - actual_szsz) <= 0.05

  #TEST FORMING OF A TWO SITE RDM. JUST CHECK THAT IS HAS THE CORRECT SIZE, TRACE AND IS PSD
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
  rdm = contract(calculate_contraction_network(
    ψψ,
    mts,
    [(v, 2) for v in vs];
    verts_tn=ITensorNetwork(ITensor[ψψsplit[vp] for vp in [(v, 2) for v in vs]]),
  ))

  rdm = array((rdm * combiner(inds(rdm; plev=0)...)) * combiner(inds(rdm; plev=1)...))
  rdm /= tr(rdm)

  eigs = eigvals(rdm)
  @test all(>=(0), real(eigs)) && all(==(0), imag(eigs))
  @test size(rdm) == (2^length(vs), 2^length(vs))

  #TEST MORE ADVANCED BP WITH ITENSORNETWORK MESSAGE TENSORS. IN THIS CASE IT SHOULD BE LIKE BOUNDARY MPS
  dims = (6, 6)
  g = named_grid(dims)
  s = siteinds("S=1/2", g)
  χ = 2
  ψ = randomITensorNetwork(s; link_space=χ)
  maxdim = 16
  v = (2, 2)
  
  ψψ = flatten_networks(ψ, dag(ψ); combine_linkinds = false, map_bra_linkinds = prime)
  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  ψOψ = flatten_networks(ψ, dag(Oψ); combine_linkinds = false, map_bra_linkinds = prime)
  
  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)
  ψOψ = combine_linkinds(ψOψ, combiners)

  
  nsites = 1
  vertex_groups = nested_graph_leaf_vertices(partition(ψψ, group(v -> v[1], vertices(ψψ))))
  mts = compute_message_tensors(
    ψψ;
    vertex_groups=vertex_groups,
    contract_kwargs=(;
      alg="density_matrix", output_structure=path_graph_structure, cutoff=1e-16, maxdim
    ),
  )
  numerator_network = calculate_contraction_network(
    ψψ, mts, [v]; verts_tn=ITensorNetwork(ψOψ[v])
  )
  
  denominator_network = calculate_contraction_network(ψψ, mts, [v])
  bp_sz = contract(numerator_network)[] / contract(denominator_network)[]
  
  exact_sz =
    contract_boundary_mps(ψOψ; cutoff=1e-16) /
    contract_boundary_mps(ψψ; cutoff=1e-16)

  @test abs.(bp_sz - exact_sz) <= 1e-2
end
