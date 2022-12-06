function ITensors.prime(indices::Array{<:Index,1}, network::Array{ITensor}, n::Integer=1)
  function primeinds(tensor)
    prime_inds = [ind for ind in inds(tensor) if ind in indices]
    if (length(prime_inds) == 0)
      return tensor
    end
    return replaceinds(tensor, prime_inds => prime(prime_inds, n))
  end
  return map(x -> primeinds(x), network)
end

function ITensors.replaceinds(
  network::Union{Array{ITensor},Array{OrthogonalITensor}}, sim_dict::Dict
)
  if length(network) == 0
    return network
  end
  indices = collect(keys(sim_dict))
  function siminds(tensor)
    sim_inds = [ind for ind in inds(tensor) if ind in indices]
    if (length(sim_inds) == 0)
      return tensor
    end
    outinds = map(i -> sim_dict[i], sim_inds)
    return replaceinds(tensor, sim_inds => outinds)
  end
  return map(x -> siminds(x), network)
end

function ITensors.commoninds(n1::Array{ITensor}, n2::Array{ITensor})
  return mapreduce(a -> commoninds(a...), vcat, zip(n1, n2))
end
