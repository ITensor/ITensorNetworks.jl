using ITensorNetworks:
  BoundaryMPSCache,
  BeliefPropagationCache,
  QuadraticFormNetwork,
  IndsNetwork,
  siteinds,
  ttn,
  random_tensornetwork,
  partitionedges,
  messages,
  update,
  partition_update,
  set_messages,
  message,
  planargraph_partitionedges,
  switch_messages,
  environment,
  VidalITensorNetwork,
  ITensorNetwork,
  expect,
  default_message_update,
  contraction_sequence,
  insert_linkinds,
  partitioned_tensornetwork,
  default_message,
  biorthogonalize,
  pe_above
using OMEinsumContractionOrders
using ITensorNetworks.ITensorsExtensions: map_eigvals
using ITensorNetworks.ModelHamiltonians: ising
using ITensors:
  ITensor,
  ITensors,
  Index,
  OpSum,
  terms,
  sites,
  contract,
  commonind,
  replaceind,
  replaceinds,
  prime,
  dag,
  noncommonind,
  noncommoninds,
  inds
using NamedGraphs: NamedGraphs, AbstractGraph, NamedEdge, NamedGraph, vertices, neighbors
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions: rem_vertex, add_edges, add_edge
using NamedGraphs.PartitionedGraphs:
  PartitionedGraph,
  partitioned_graph,
  PartitionVertex,
  PartitionEdge,
  unpartitioned_graph,
  partitioned_vertices,
  which_partition
using LinearAlgebra: normalize
using Graphs: center

using Random

Random.seed!(1834)
ITensors.disable_warn_order()

function main()
  L = 4
  #g = lieb_lattice_grid(L, L)
  #g = named_hexagonal_lattice_graph(L, L)
  #g = named_grid_periodic_x((L,2))
  g = named_grid((L, 3))
  vc = first(center(g))
  s = siteinds("S=1/2", g)
  ψ = random_tensornetwork(s; link_space=3)
  bp_update_kwargs = (; maxiter=50, tol=1e-14, verbose=true)

  ψIψ = BeliefPropagationCache(QuadraticFormNetwork(ψ))
  ψIψ = update(ψIψ; bp_update_kwargs...)
  return ψIψ = update(ψIψ; bp_update_kwargs...)
end

main()
