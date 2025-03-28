using ITensorNetworks:
  BoundaryMPSCache,
  BeliefPropagationCache,
  QuadraticFormNetwork,
  IndsNetwork,
  siteinds,
  ttn,
  inner,
  random_tensornetwork,
  partitionedges,
  messages,
  update,
  partition_update,
  set_messages,
  message,
  virtual_index_dimension,
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
  partitions,
  partitionpairs,
  region_scalar,
  scalar_factors_quotient
using OMEinsumContractionOrders
using ITensorNetworks.ITensorsExtensions: map_eigvals
using ITensorNetworks.ModelHamiltonians: ising
using ITensors:
  Algorithm,
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
  inds,
  scalar
using NamedGraphs: NamedGraphs, AbstractGraph, NamedEdge, NamedGraph, vertices, neighbors
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions:
  rem_vertex, add_edges, add_edge, rem_vertices, add_vertices
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

function lieb_lattice_grid(nx::Int64, ny::Int64; periodic=false)
  @assert (!periodic && isodd(nx) && isodd(ny)) || (periodic && iseven(nx) && iseven(ny))
  g = named_grid((nx, ny); periodic)
  for v in vertices(g)
    if iseven(first(v)) && iseven(last(v))
      g = rem_vertex(g, v)
    end
  end
  return g
end

function named_grid_periodic_x(nxny::Tuple)
  nx, ny = nxny
  g = named_grid((nx, ny))
  for i in 1:ny
    g = add_edge(g, NamedEdge((nx, i) => (1, i)))
  end
  return g
end

function exact_expect(ψ::ITensorNetwork, ops::Vector{<:String}, vs::Vector)
  s = siteinds(ψ)
  ψIψ = QuadraticFormNetwork(ψ)
  ψOψ = QuadraticFormNetwork(ψ)
  for (op_string, v) in zip(ops, vs)
    ψOψ[(v, "operator")] = ITensors.op(op_string, s[v])
  end
  numer_seq = contraction_sequence(ψOψ; alg="sa_bipartite")
  denom_seq = contraction_sequence(ψIψ; alg="sa_bipartite")
  numer, denom = contract(ψOψ; sequence=numer_seq)[], contract(ψIψ; sequence=denom_seq)[]
  return numer / denom
end

function exact_expect(ψ::ITensorNetwork, op_string::String, v)
  return exact_expect(ψ, [op_string], [v])
end

function make_eigs_real(A::ITensor)
  return map_eigvals(x -> real(x), A, first(inds(A)), last(inds(A)); ishermitian=true)
end

Random.seed!(1834)
ITensors.disable_warn_order()

function main()
  L = 4
  #g = named_grid((L, 2))
  g = named_hexagonal_lattice_graph(3, 3)

  vc = first(center(g))
  s = siteinds("S=1/2", g)
  ψ = random_tensornetwork(s; link_space=2)
  bp_update_kwargs = (;
    maxiter=50,
    tol=1e-14,
    message_update_kwargs=(;
      message_update_function=ms -> make_eigs_real.(default_message_update(ms))
    ),
  )
  #Run BP first to normalize and put in a stable gauge
  ψIψ = BeliefPropagationCache(QuadraticFormNetwork(ψ))
  ψIψ = update(ψIψ; bp_update_kwargs...)
  ψ = VidalITensorNetwork(
    ψ; (cache!)=Ref(ψIψ), update_cache=false, cache_update_kwargs=(; maxiter=0)
  )
  ψ = ITensorNetwork(ψ)

  ψIψ_bp = BeliefPropagationCache(QuadraticFormNetwork(ψ))
  ψIψ_boundarymps = BoundaryMPSCache(ψIψ; message_rank=2)
  @show ITensorNetwork(ψIψ_boundarymps, 1 => 2)
end

main()
