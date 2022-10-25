mutable struct TensorNetworkGraph
  network::Vector{ITensor}
  graph::SimpleDiGraph
  weights::Matrix
  #a dict that maps uncontracted Index to the adjacent vertices pair (i,j)
  out_edge_dict::Dict
  inner_edge_dict::Dict
end

function TensorNetworkGraph(network::Vector{ITensor})
  uncontract_inds = noncommoninds(network...)
  graph = Graphs.DiGraph(length(network))
  # construct contract_edges
  contract_edges = []
  inner_edge_dict = Dict()
  for (i, t) in enumerate(network)
    for ind in setdiff(inds(t), uncontract_inds)
      if !haskey(inner_edge_dict, [ind])
        inner_edge_dict[[ind]] = (i, log2(space(ind)))
      else
        @assert(length(inner_edge_dict[[ind]]) == 2)
        inner_edge_dict[[ind]] = (inner_edge_dict[[ind]][1], i, inner_edge_dict[[ind]][2])
        push!(contract_edges, inner_edge_dict[[ind]])
      end
    end
  end
  weights = zeros(length(network), length(network))
  for e in contract_edges
    u, v, f = e
    Graphs.add_edge!(graph, u, v)
    Graphs.add_edge!(graph, v, u)
    weights[u, v] = f
    weights[v, u] = f
  end
  # construct out_edge_dict
  out_edge_dict = Dict()
  for (i, t) in enumerate(network)
    ucinds = intersect(inds(t), uncontract_inds)
    if length(ucinds) == 0
      continue
    end
    for ind in ucinds
      out_edge_dict[[ind]] = (i, log2(space(ind)))
    end
  end
  return TensorNetworkGraph(network, graph, weights, out_edge_dict, inner_edge_dict)
end

Base.show(io::IO, tng::TensorNetworkGraph) = print(io, tng.out_edge_dict)

function Base.copy(tng::TensorNetworkGraph)
  return TensorNetworkGraph(
    tng.network,
    copy(tng.graph),
    copy(tng.weights),
    copy(tng.out_edge_dict),
    copy(tng.inner_edge_dict),
  )
end

function ITensors.noncommoninds(tng::TensorNetworkGraph)
  return keys(tng.out_edge_dict)
end

function distance(tng::TensorNetworkGraph, s, t)
  sindex = tng.out_edge_dict[s][1]
  ds = dijkstra_shortest_paths(tng.graph, sindex, tng.weights)
  get_dist(edge) = ds.dists[tng.out_edge_dict[edge][1]]
  return get_dist(t)
end

#TODO: delete this
distance(tng::TensorNetworkGraph, s) = 0.0

function insert_outedge_vertex!(tng::TensorNetworkGraph)
  if length(tng.out_edge_dict) == 0
    return nothing
  end
  add_vertices!(tng.graph, 1)
  t = size(tng.graph)[1]
  new_weights = zeros(t, t)
  new_weights[1:(t - 1), 1:(t - 1)] = tng.weights
  for (inds, edge) in tng.out_edge_dict
    u, wu = edge
    Graphs.add_edge!(tng.graph, u, t)
    Graphs.add_edge!(tng.graph, t, u)
    new_weights[u, t] = wu
    new_weights[t, u] = wu
    tng.out_edge_dict[inds] = (u, t, wu)
  end
  return tng.weights = new_weights
end

function indsname(inds::Vector)
  if length(inds) == 1
    return string(inds[1].tags)
  end
  return ""
end

function visualize(tng::TensorNetworkGraph)
  tng = copy(tng)
  insert_outedge_vertex!(tng::TensorNetworkGraph)
  wg = SimpleWeightedGraph(tng.graph)
  for e in edges(tng.graph)
    add_edge!(wg, src(e), dst(e), tng.weights[src(e), dst(e)])
  end
  edgelabel_dict = Dict{Tuple{Int,Int},String}()
  edgecolor_dict = Dict()
  for (inds, edge) in tng.inner_edge_dict
    if indsname(inds) == ""
      edgelabel_dict[(edge[1], edge[2])] =
        indsname(inds) * "w=" * string(round(edge[3]; digits=2))
      edgecolor_dict[(edge[1], edge[2])] = :blue
    else
      edgelabel_dict[(edge[1], edge[2])] =
        indsname(inds) * "w=" * string(round(edge[3]; digits=2))
      edgecolor_dict[(edge[1], edge[2])] = :black
    end
  end
  for (inds, edge) in tng.out_edge_dict
    edgelabel_dict[(edge[1], edge[2])] =
      indsname(inds) * "w=" * string(round(edge[3]; digits=2))
    edgecolor_dict[(edge[1], edge[2])] = :red
  end
  return graphplot(
    wg;
    markersize=0.3,
    # names=names,
    edgelabel=edgelabel_dict,
    curves=false,
    edgecolor=edgecolor_dict,
    linewidth=20,
    fontsize=30,
    size=(7000, 7000),
  )
end
