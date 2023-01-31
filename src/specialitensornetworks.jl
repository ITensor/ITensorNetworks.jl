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
BUILD Z OF CLASSICAL ISING MODEL ON A GIVEN GRAPH AT INVERSE TEMP BETA
H = -\\sum_{(v,v') \\in edges}\\sigma^{z}_{v}\\sigma^{z}_{v'}
OPTIONAL ARGUMENT:
  h: EXTERNAL MAGNETIC FIELD
  szverts: A LIST OF VERTICES OVER WHICH TO APPLY A SZ.
    THE RESULTANT NETWORK CAN THEN BE CONTRACTED AND DIVIDED BY THE ACTUAL PARTITION FUNCTION TO GET THAT OBSERVABLE
    INDSNETWORK IS ASSUMED TO BE BUILT FROM A GRAPH (NO SITE INDS) AND OF LINK SPACE 2
"""
function ising_network(
  eltype::Type, s::IndsNetwork, beta::Number; h::Number=0.0, szverts=nothing
)
  tn = delta_network(eltype, s)
  if (szverts != nothing)
    for v in szverts
      tn[v] = diagITensor(eltype[1, -1], inds(tn[v]))
    end
  end
  for edge in edges(tn)
    v1 = src(edge)
    v2 = dst(edge)
    i = commoninds(tn[v1], tn[v2])[1]
    deg_v1 = degree(tn, v1)
    deg_v2 = degree(tn, v2)
    f11 = exp(beta * (1 + h / deg_v1 + h / deg_v2))
    f12 = exp(beta * (-1 + h / deg_v1 - h / deg_v2))
    f21 = exp(beta * (-1 - h / deg_v1 + h / deg_v2))
    f22 = exp(beta * (1 - h / deg_v1 - h / deg_v2))
    q = eltype[f11 f12; f21 f22]
    w, V = eigen(q)
    w = map(sqrt, w)
    sqrt_q = V * ITensors.Diagonal(w) * inv(V)
    t = itensor(sqrt_q, i, i')
    tn[v1] = tn[v1] * t
    tn[v1] = noprime!(tn[v1])
    t = itensor(sqrt_q, i', i)
    tn[v2] = tn[v2] * t
    tn[v2] = noprime!(tn[v2])
  end
  return tn
end

function ising_network(s::IndsNetwork, beta::Number; h::Number=0.0, szverts=nothing)
  return ising_network(typeof(beta), s, beta; h, szverts)
end

"""
BUILD Z OF CLASSICAL ISING MODEL ON A GIVEN GRAPH AT INVERSE TEMP BETA
H = -\\sum_{(v,v') \\in edges}\\sigma^{z}_{v}\\sigma^{z}_{v'}
TAKE AS AN OPTIONAL ARGUMENT A LIST OF VERTICES OVER WHICH TO APPLY A SZ. THE RESULTANT NETWORK CAN THEN BE CONTRACTED AND DIVIDED BY THE ACTUAL PARTITION FUNCTION TO GET THAT OBSERVABLE
"""
function ising_network(
  eltype::Type, g::NamedGraph, beta::Number; h::Number=0.0, szverts=nothing
)
  return ising_network(eltype, IndsNetwork(g; link_space=2), beta; h, szverts)
end

function ising_network(g::NamedGraph, beta::Number; h::Number=0.0, szverts=nothing)
  return ising_network(eltype(beta), g, beta; h, szverts)
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

@traitfn function randomITensorNetwork(
  eltype::Type, g::::IsUnderlyingGraph; link_space=nothing
)
  return randomITensorNetwork(eltype, IndsNetwork(g); link_space)
end

@traitfn function randomITensorNetwork(g::::IsUnderlyingGraph; link_space=nothing)
  return randomITensorNetwork(Float64, IndsNetwork(g); link_space)
end
