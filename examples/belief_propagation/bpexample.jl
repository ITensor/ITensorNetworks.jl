using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine

using ITensorNetworks:
  belief_propagation, approx_network_region, contract_inner, message_tensors

function main()
  n = 4
  g_dims = (n, n)
  g = named_grid(g_dims)
  s = siteinds("S=1/2", g)
  chi = 2

  Random.seed!(5467)

  #bra
  ψ = randomITensorNetwork(s; link_space=chi)
  #bra-ket (but not actually contracted)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  #Site to take expectation value on
  v = (1, 1)

  #Now do Simple Belief Propagation to Measure Sz on Site v
  mts = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  numerator_network = approx_network_region(
    ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork([apply(op("Sz", s[v]), ψ[v])])
  )
  denominator_network = approx_network_region(ψψ, mts, [(v, 1)])
  sz_bp = contract(numerator_network)[] / contract(denominator_network)[]

  println(
    "Simple Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_bp)
  )

  #Now do General Belief Propagation to Measure Sz on Site v
  nsites = 4
  Zp = partition(
    partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=nsites
  )
  Zpp = partition(ψψ; subgraph_vertices=nested_graph_leaf_vertices(Zp))
  mts = message_tensors(Zpp)
  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  numerator_network = approx_network_region(
    ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork([apply(op("Sz", s[v]), ψ[v])])
  )
  denominator_network = approx_network_region(ψψ, mts, [(v, 1)])
  sz_bp = contract(numerator_network)[] / contract(denominator_network)[]

  println(
    "General Belief Propagation (4-site subgraphs) Gives Sz on Site " *
    string(v) *
    " as " *
    string(sz_bp),
  )

  #Now do General Belief Propagation with Matrix Product State Message Tensors Measure Sz on Site v
  ψψ = flatten_networks(ψ, dag(ψ); combine_linkinds=false, map_bra_linkinds=prime)
  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  ψOψ = flatten_networks(ψ, dag(Oψ); combine_linkinds=false, map_bra_linkinds=prime)

  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)
  ψOψ = combine_linkinds(ψOψ, combiners)

  Z = partition(ψψ, group(v -> v[1], vertices(ψψ)))
  maxdim = 8
  mts = message_tensors(Z)

  # mts = belief_propagation(
  #   ψψ,
  #   mts;
  #   contract_kwargs=(;
  #     alg="density_matrix",
  #     output_structure=path_graph_structure,
  #     maxdim,
  #     contraction_sequence_alg="optimal",
  #   ),
  # )

  mts = belief_propagation(ψψ, mts; contract_kwargs=(; alg="exact"))
  numerator_network = approx_network_region(ψψ, mts, [v]; verts_tn=ITensorNetwork(ψOψ[v]))
  denominator_network = approx_network_region(ψψ, mts, [v])
  sz_bp = contract(numerator_network)[] / contract(denominator_network)[]

  println(
    "General Belief Propagation with Column Partitioning and MPS Message Tensors (Max dim 8) Gives Sz on Site " *
    string(v) *
    " as " *
    string(sz_bp),
  )

  #Now do it exactly
  sz_exact = contract(ψOψ)[] / contract(ψψ)[]

  return println("The exact value of Sz on Site " * string(v) * " is " * string(sz_exact))
end

main()
