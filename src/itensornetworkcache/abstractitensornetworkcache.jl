abstract type AbstractITensorNetworkCache end
const AbstractITNCache = AbstractITensorNetworkCache

# Create a cache from an ITensorNetwork.
function cache(tn::AbstractITensorNetwork, vs::Vector, in_vs::Function, out_vs::Function, inds_map::Function; contract_alg)
  return cache(Algorithm(contract_alg), tn, vs, in_vs, out_vs, inds_map)
end
