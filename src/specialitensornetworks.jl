"""
RETURN A TENSOR NETWORK WITH COPY TENSORS ON EACH VERTEX. 
Note that passing a link_space will mean the indices of the resulting network don't match those of the input indsnetwork
"""
function delta_network(eltype::Type, s::IndsNetwork; link_space=nothing)
  return ITensorNetwork((v, inds...) -> δ(eltype, inds...), s; link_space)
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
BUILD Z OF CLASSICAL ISING MODEL ON A GIVEN GRAPH AT INVERSE TEMP BETA
H = -\\sum_{(v,v') \\in edges}\\sigma^{z}_{v}\\sigma^{z}_{v'}
TAKE AS AN OPTIONAL ARGUMENT A LIST OF VERTICES OVER WHICH TO APPLY A SZ. THE RESULTANT NETWORK CAN THEN BE CONTRACTED AND DIVIDED BY THE ACTUAL PARTITION FUNCTION TO GET THAT OBSERVABLE
INDSNETWORK IS ASSUMED TO BE BUILT FROM A GRAPH (NO SITE INDS) AND OF LINK SPACE 2
"""
function ising_network(eltype::Type, s::IndsNetwork, beta::Number; szverts=nothing)
  ψ = delta_network(eltype, s)

  if (szverts != nothing)
    for v in szverts
      ψ[v] = diagITensor(eltype[1, -1], inds(ψ[v]))
    end
  end
  J = 1
  f1 = 0.5 * sqrt(exp(-J * 2 * beta * 0.5) * (-1 + exp(J * 2 * beta)))
  f2 = 0.5 * sqrt(exp(-J * 2 * beta * 0.5) * (1 + exp(J * 2 * beta)))
  q = eltype[(f1+f2) (-f1+f2); (-f1+f2) (f1+f2)]
  for v in vertices(ψ)
    is = inds(ψ[v])
    indices = inds(ψ[v])
    for i in indices
      qtens = itensor(q, i, i')
      ψ[v] = ψ[v] * qtens
      ψ[v] = noprime!(ψ[v])
    end
  end

  return ψ
end

function ising_network(s::IndsNetwork, beta::Number; szverts=nothing)
  return ising_network(typeof(beta), s; szverts)
end

"""
BUILD Z OF CLASSICAL ISING MODEL ON A GIVEN GRAPH AT INVERSE TEMP BETA
H = -\\sum_{(v,v') \\in edges}\\sigma^{z}_{v}\\sigma^{z}_{v'}
TAKE AS AN OPTIONAL ARGUMENT A LIST OF VERTICES OVER WHICH TO APPLY A SZ. THE RESULTANT NETWORK CAN THEN BE CONTRACTED AND DIVIDED BY THE ACTUAL PARTITION FUNCTION TO GET THAT OBSERVABLE
"""
function ising_network(eltype::Type, g::NamedGraph, beta::Number; szverts=nothing)
  return ising_network(eltype, IndsNetwork(g; link_space=2), beta; szverts)
end

function ising_network(g::NamedGraph, beta::Number; szverts=nothing)
  return ising_network(eltype(beta), g, beta; szverts)
end

"""
Build an ITensor network on a graph specified by the inds network s. Bond_dim is given by link_space and entries are randomised (normal distribution, mean 0 std 1)
"""
function randomITensorNetwork(eltype::Type, s::IndsNetwork; link_space=nothing)
  return ITensorNetwork(s; link_space) do v, inds...
    itensor(randn(eltype, dim(inds)...), inds...)
  end
end

function randomITensorNetwork(s::IndsNetwork; link_space=nothing)
  return randomITensorNetwork(Float64, s; link_space)
end

@traitfn function randomITensorNetwork(eltype::Type, g::::IsUnderlyingGraph; link_space=nothing)
  return randomITensorNetwork(eltype, IndsNetwork(g); link_space)
end

@traitfn function randomITensorNetwork(g::::IsUnderlyingGraph; link_space=nothing)
  return randomITensorNetwork(Float64, IndsNetwork(g); link_space)
end
