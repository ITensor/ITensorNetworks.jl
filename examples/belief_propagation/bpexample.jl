using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine

using ITensorNetworks:
  compute_message_tensors,
  calculate_contraction_network,
  contract_inner,
  nested_graph_leaf_vertices

function main()
  n = 4
  dims = (n, n)
  g = named_grid(dims)
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
  nsites = 1

  vertex_groups = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=nsites)
  )
  mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)
  numerator_network = calculate_contraction_network(
    ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork([apply(op("Sz", s[v]), ψ[v])])
  )
  denominator_network = calculate_contraction_network(ψψ, mts, [(v, 1)])
  sz_bp = contract(numerator_network)[] / contract(denominator_network)[]

  println(
    "Simple Belief Propagation Gives Sz on Site " * string(v) * " as " * string(sz_bp)
  )

  #Now do General Belief Propagation to Measure Sz on Site v
  nsites = 4
  vertex_groups = nested_graph_leaf_vertices(
    partition(partition(ψψ, group(v -> v[1], vertices(ψψ))); nvertices_per_partition=nsites)
  )
  mts = compute_message_tensors(ψψ; vertex_groups=vertex_groups)
  numerator_network = calculate_contraction_network(
    ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork([apply(op("Sz", s[v]), ψ[v])])
  )
  denominator_network = calculate_contraction_network(ψψ, mts, [(v, 1)])
  sz_bp = contract(numerator_network)[] / contract(denominator_network)[]

  println(
    "General Belief Propagation (4-site subgraphs) Gives Sz on Site " *
    string(v) *
    " as " *
    string(sz_bp),
  )

  #Now do General Belief Propagation with Matrix Product State Message Tensors Measure Sz on Site v
  vertex_groups = nested_graph_leaf_vertices(
    partition(ψψ, group(v -> v[1][1], vertices(ψψ)))
  )
  maxdim = 8

  mts = compute_message_tensors(
    ψψ;
    vertex_groups=vertex_groups,
    contract_kwargs=(;
      alg="density_matrix",
      output_structure=path_graph_structure,
      maxdim,
      contraction_sequence_alg="greedy",
    ),
  )
  numerator_network = calculate_contraction_network(
    ψψ, mts, [(v, 1)]; verts_tn=ITensorNetwork([apply(op("Sz", s[v]), ψ[v])])
  )
  denominator_network = calculate_contraction_network(ψψ, mts, [(v, 1)])
  contract_sequence = contraction_sequence(numerator_network; alg="greedy")
  sz_bp =
    contract(numerator_network; sequence=contract_sequence)[] /
    contract(denominator_network; sequence=contract_sequence)[]

  println(
    "General Belief Propagation with Column Partitioning and MPS Message Tensors (Max dim 8) Gives Sz on Site " *
    string(v) *
    " as " *
    string(sz_bp),
  )

  #Now do it exactly
  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  sz_exact = contract_inner(Oψ, ψ) / contract_inner(ψ, ψ)

  return println("The exact value of Sz on Site " * string(v) * " is " * string(sz_exact))
end

main()
