using Compat
using ITensors
using Metis
using ITensorNetworks
using Random
using SplitApplyCombine
using Graphs
using NamedGraphs

using ITensorNetworks:
  belief_propagation,
  approx_network_region,
  contract_inner,
  message_tensors,
  nested_graph_leaf_vertices

function main()
  g = named_comb_tree((6, 6))
  s = siteinds("S=1/2", g)
  χ = 4

  Random.seed!(5467)

  ψ = randomITensorNetwork(s; link_space=χ)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  #Initial message tensors for BP
  mts_init = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  println("\nFirst testing out a comb tree. Random network with bond dim $χ")

  #Now test out various sequences
  print("Parallel updates (sequence is irrelevant): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    edges=[[e] for e in edges(mts_init)],
    verbose=true,
  )
  print("Sequential updates (sequence is default edge list of the message tensors): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    edges=[e for e in edges(mts_init)],
    verbose=true,
  )
  print("Sequential updates (sequence is our custom sequence finder): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    verbose=true,
  )

  g = NamedGraph(Graphs.random_regular_graph(100, 3))
  s = siteinds("S=1/2", g)
  χ = 4

  Random.seed!(5467)

  ψ = randomITensorNetwork(s; link_space=χ)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  #Initial message tensors for BP
  mts_init = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  println("\nNow testing out a z = 3 random regular graph. Random network with bond dim $χ")

  #Now test out various sequences
  print("Parallel updates (sequence is irrelevant): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    edges=[[e] for e in edges(mts_init)],
    verbose=true,
  )
  print("Sequential updates (sequence is default edge list of the message tensors): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    edges=[e for e in edges(mts_init)],
    verbose=true,
  )
  print("Sequential updates (sequence is our custom sequence finder): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    verbose=true,
  )

  g = named_grid((6, 6))
  s = siteinds("S=1/2", g)
  χ = 2

  Random.seed!(5467)

  ψ = randomITensorNetwork(s; link_space=χ)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  #Initial message tensors for BP
  mts_init = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  println("\nNow testing out a 6x6 grid. Random network with bond dim $χ")

  #Now test out various sequences
  print("Parallel updates (sequence is irrelevant): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    edges=[[e] for e in edges(mts_init)],
    verbose=true,
  )
  print("Sequential updates (sequence is default edge list of the message tensors): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    edges=[e for e in edges(mts_init)],
    verbose=true,
  )
  print("Sequential updates (sequence is our custom sequence finder): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    verbose=true,
  )

  g = NamedGraphs.hexagonal_lattice_graph(4, 4)
  s = siteinds("S=1/2", g)
  χ = 3

  Random.seed!(5467)

  ψ = randomITensorNetwork(s; link_space=χ)
  ψψ = ψ ⊗ prime(dag(ψ); sites=[])

  #Initial message tensors for BP
  mts_init = message_tensors(
    ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
  )

  println("\nNow testing out a 4 x 4 hexagonal lattice. Random network with bond dim $χ")

  #Now test out various sequences
  print("Parallel updates (sequence is irrelevant): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    edges=[[e] for e in edges(mts_init)],
    verbose=true,
  )
  print("Sequential updates (sequence is default edge list of the message tensors): ")
  belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    edges=[e for e in edges(mts_init)],
    verbose=true,
  )
  print("Sequential updates (sequence is our custom sequence finder): ")
  return belief_propagation(
    ψψ,
    mts_init;
    contract_kwargs=(; alg="exact"),
    target_precision=1e-10,
    niters=100,
    verbose=true,
  )
end

main()
