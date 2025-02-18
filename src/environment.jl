using ITensors: contract
using NamedGraphs.PartitionedGraphs: PartitionedGraph

default_environment_algorithm() = "bp"

function environment(
  tn::AbstractITensorNetwork,
  vertices::Vector;
  alg=default_environment_algorithm(),
  kwargs...,
)
  return environment(Algorithm(alg), tn, vertices; kwargs...)
end

function environment(
  ::Algorithm"exact", tn::AbstractITensorNetwork, verts::Vector; kwargs...
)
  return [contract(subgraph(tn, setdiff(vertices(tn), verts)); kwargs...)]
end

function environment(
  alg::Algorithm,
  ptn::PartitionedGraph,
  vertices::Vector;
  (cache!)=nothing,
  update_cache=isnothing(cache!),
  cache_construction_kwargs=default_cache_construction_kwargs(alg, ptn),
  cache_update_kwargs=default_cache_update_kwargs(alg),
)
  if isnothing(cache!)
    cache! = Ref(cache(alg, ptn; cache_construction_kwargs...))
  end

  if update_cache
    cache![] = update(cache![]; cache_update_kwargs...)
  end

  return environment(cache![], vertices)
end

function environment(
  alg::Algorithm,
  tn::AbstractITensorNetwork,
  vertices::Vector;
  partitioned_vertices=default_partitioned_vertices(tn),
  kwargs...,
)
  return environment(alg, PartitionedGraph(tn, partitioned_vertices), vertices; kwargs...)
end
