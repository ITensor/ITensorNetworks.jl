# TODO: Define `ITensorNetworkMap` that is an `ITensorNetwork` with an `inds_map`.
struct BPCache{TN,Cache,V,In,Out,Map} <: AbstractITNCache
  tn::TN # ITensorNetwork (unpartitioned? of quadratic form)
  cache::Cache # DataGraph of message tensors
  vs::Vector{V} # Vertices of original tensor network state
  update_region::Vector{V} # Region to update
  in_vs::In # Bra vertices of tensor network state
  out_vs::Out # Ket vertices of tensor network state
  inds_map::Map # Map indices from ket to bra
end

set_cache(cache::BPCache, new_cache) = @set cache.cache = new_cache

function cache(contract_alg::Algorithm"bp", tn::AbstractITensorNetwork, vs::Vector, in_vs::Function, out_vs::Function, inds_map::Function)
  update_region = eltype(vs)[]
  return BPCache(tn, DataGraph(), vs, update_region, in_vs, out_vs, inds_map)
end

# Apply the quadratic form to an input on the region
# implicitly defined in the cache.
function (cache::BPCache)(v)
  @show cache.update_region
  error("Not implemented")
end

# TODO: Delete, I don't think this is needed since it is
# implicit from the length of `cache.update_region`.
function set_nsite(cache::BPCache, nsite)
  error("Not implemented")
end

function position(cache::BPCache, tn::AbstractITensorNetwork, region)
  # A partitioning of the vertices for the BP cache
  vertex_partitions = map(v -> vertex_path(cache.tn, cache.in_vs(v), cache.out_vs(v)), Indices(cache.vs))
  tn_partitioned = partition(cache.tn, vertex_partitions)

  # TODO: `belief_propagation(tn, vertex_partitions) -> mts`
  # where `mts` is stored in an `EdgeDataGraph` and `vertex_partitions`
  # is a `Dictionary` of vertex partitions.
  mts = message_tensors(tn_partitioned)
  mts = belief_propagation(cache.tn, mts; contract_kwargs=(; alg="exact"))

  # TODO: Set the `update_region`, `set_update_region`.
  return set_cache(cache, mts)
end
