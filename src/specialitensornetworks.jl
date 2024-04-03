using ITensors: delta
using ITensors.NDTensors: dim
using DataGraphs: IsUnderlyingGraph
using Distributions: Distribution

"""
RETURN A TENSOR NETWORK WITH COPY TENSORS ON EACH VERTEX. 
Note that passing a link_space will mean the indices of the resulting network don't match those of the input indsnetwork
"""
function delta_network(eltype::Type, s::IndsNetwork; link_space=nothing)
  return ITensorNetwork((v, inds...) -> delta(eltype, inds...), s; link_space)
end

function delta_network(s::IndsNetwork; link_space=nothing)
  return delta_network(Float64, s; link_space)
end

function delta_network(eltype::Type, graph::AbstractNamedGraph; link_space=nothing)
  return delta_network(eltype, IndsNetwork(graph; link_space))
end

function delta_network(graph::AbstractNamedGraph; link_space=nothing)
  return delta_network(Float64, graph; link_space)
end

"""
Build an ITensor network on a graph specified by the inds network s. Bond_dim is given by link_space and entries are randomised (normal distribution, mean 0 std 1)
"""
function random_itensornetwork(eltype::Type, s::IndsNetwork; link_space=nothing)
  return ITensorNetwork(s; link_space) do v, inds...
    itensor(randn(eltype, dim(inds)...), inds...)
  end
end

function random_itensornetwork(s::IndsNetwork; link_space=nothing)
  return random_itensornetwork(Float64, s; link_space)
end

@traitfn function random_itensornetwork(
  eltype::Type, g::::IsUnderlyingGraph; link_space=nothing
)
  return random_itensornetwork(eltype, IndsNetwork(g); link_space)
end

@traitfn function random_itensornetwork(g::::IsUnderlyingGraph; link_space=nothing)
  return random_itensornetwork(Float64, IndsNetwork(g); link_space)
end

"""
Build an ITensor network on a graph specified by the inds network s.
Bond_dim is given by link_space and entries are randomized.
The random distribution is based on the input argument `distribution`.
"""
function random_itensornetwork(
  distribution::Distribution, s::IndsNetwork; link_space=nothing
)
  return ITensorNetwork(s; link_space) do v, inds...
    itensor(rand(distribution, dim(inds)...), inds...)
  end
end

@traitfn function random_itensornetwork(
  distribution::Distribution, g::::IsUnderlyingGraph; link_space=nothing
)
  return random_itensornetwork(distribution, IndsNetwork(g); link_space)
end
