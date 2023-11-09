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
  nested_graph_leaf_vertices,
  edge_sequence

function main()
  g_labels = [
    "Comb Tree",
    "100 Site Random Regular Graph z = 3",
    "6x6 Square Grid",
    "4x4 Hexagonal Lattice",
  ]
  gs = [
    named_comb_tree((6, 6)),
    NamedGraph(Graphs.random_regular_graph(100, 3)),
    named_grid((6, 6)),
    NamedGraphs.hexagonal_lattice_graph(4, 4),
  ]
  χs = [4, 4, 2, 3]

  for (i, g) in enumerate(gs)
    Random.seed!(5467)
    g_label = g_labels[i]
    χ = χs[i]
    s = siteinds("S=1/2", g)
    ψ = randomITensorNetwork(s; link_space=χ)
    ψψ = ψ ⊗ prime(dag(ψ); sites=[])

    #Initial message tensors for BP
    mts_init = message_tensors(
      ψψ; subgraph_vertices=collect(values(group(v -> v[1], vertices(ψψ))))
    )

    println("\nFirst testing out a $g_label. Random network with bond dim $χ")

    #Now test out various sequences
    print("Parallel updates (sequence is irrelevant): ")
    belief_propagation(
      ψψ,
      mts_init;
      contract_kwargs=(; alg="exact"),
      target_precision=1e-10,
      niters=100,
      edges=edge_sequence(mts_init; alg="parallel"),
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
  end
end

main()
