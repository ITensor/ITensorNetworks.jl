using ITensorNetworks: random_tensornetwork, indsnetwork, expect, contraction_sequence
using NamedGraphs: NamedGraph, NamedEdge, position_graph, vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_hexagonal_lattice_graph
using NamedGraphs.GraphsExtensions:
  induced_subgraph, neighbors, add_vertices, add_edges, add_edge, add_edge!, add_vertex!
using NamedGraphs.PartitionedGraphs: PartitionEdge
using Graphs: has_edge
using ITensors: contract, siteinds
using EinExprs: Exhaustive, Greedy, HyPar
using OMEinsumContractionOrders
using Dictionaries

using TensorOperations: optimaltree

using ITensors: Index, inds, id, dim

function indices_to_symbol(inds::Vector{<:Index})
  d = Dictionary()
  d_dim = Dictionary()
  inds_set = Index[]
  count = 1
  for ind in inds
    if ind ∉ inds_set
      set!(d, ind, Symbol(count))
      set!(d_dim, Symbol(count), dim(ind))
      count += 1
    end
  end
  d_dim = Dict(k => d_dim[k] for k in collect(keys(d_dim)))
  return d, d_dim
end

function optimal_seq_to(tn::ITensorNetwork)
  all_inds = unique(reduce(vcat, [[i for i in inds(ψ[v])] for v in vertices(ψ)]))
  inds_to_symbols, symbols_to_dims = indices_to_symbol(all_inds)

  network = [[inds_to_symbols[i] for i in inds(ψ[v])] for v in vertices(ψ)]

  t = time()
  seq = optimaltree(network, symbols_to_dims)
  t1 = time() - t
  return seq, t1
end

println("3 x 3 Tensor Network")
g = named_grid((3, 3))
ψ = random_tensornetwork(g; link_space=2)

all_inds = unique(reduce(vcat, [[i for i in inds(ψ[v])] for v in vertices(ψ)]))

t = ψ[(1, 1)]

inds_to_symbols, symbols_to_dims = indices_to_symbol(all_inds)

network = [[inds_to_symbols[i] for i in inds(ψ[v])] for v in vertices(ψ)]

seq = optimaltree(network, symbols_to_dims)
