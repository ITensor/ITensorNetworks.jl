#RETURN A TENSOR NETWORK WITH COPY TENSORS ON EACH VERTEX. 
#Note that passing a link_space will mean the indices of the resulting network don't match those of the input indsnetwork
function delta_network(s::IndsNetwork; link_space=nothing)
  ψ = ITensorNetwork(s; link_space)

  for v in vertices(ψ)
    is = inds(ψ[v])
    ψ[v] = delta(is)
  end
  return ψ
end

function delta_network(g::NamedDimGraph; link_space=2)
  s = IndsNetwork(g; link_space=link_space)
  return delta_network(s; link_space=link_space)
end

#BUILD Z OF CLASSICAL ISING MODEL ON A GIVEN GRAPH AT INVERSE TEMP BETA
#H = -\sum_{(v,v') \in edges}\sigma^{z}_{v}\sigma^{z}_{v'}
#TAKE AS AN OPTIONAL ARGUMENT A LIST OF VERTICES OVER WHICH TO APPLY A SZ. THE RESULTANT NETWORK CAN THEN BE CONTRACTED AND DIVIDED BY THE ACTUAL PARTITION FUNCTION TO GET THAT OBSERVABLE
#INDSNETWORK IS ASSUMED TO BE BUILT FROM A GRAPH (NO SITE INDS) AND OF LINK SPACE 2
function ising_network(s::IndsNetwork, beta::Float64; szverts=nothing)
  ψ = delta_network(s)

  if (szverts != nothing)
    for v in szverts
      ψ[v] = ITensors.diagITensor([1, -1], inds(ψ[v]))
    end
  end
  J = 1
  f1 = 0.5 * sqrt(exp(-J * 2 * beta * 0.5) * (-1 + exp(J * 2 * beta)))
  f2 = 0.5 * sqrt(exp(-J * 2 * beta * 0.5) * (1 + exp(J * 2 * beta)))
  q = [(f1+f2) (-f1+f2); (-f1+f2) (f1+f2)]
  for v in vertices(ψ)
    is = inds(ψ[v])
    indices = inds(ψ[v])
    for i in indices
      qtens = ITensor(q, i, i')
      ψ[v] = ψ[v] * qtens
      ψ[v] = noprime!(ψ[v])
    end
  end

  return ψ
end

#BUILD Z OF CLASSICAL ISING MODEL ON A GIVEN GRAPH AT INVERSE TEMP BETA
#H = -\sum_{(v,v') \in edges}\sigma^{z}_{v}\sigma^{z}_{v'}
#TAKE AS AN OPTIONAL ARGUMENT A LIST OF VERTICES OVER WHICH TO APPLY A SZ. THE RESULTANT NETWORK CAN THEN BE CONTRACTED AND DIVIDED BY THE ACTUAL PARTITION FUNCTION TO GET THAT OBSERVABLE
function ising_network(g::NamedDimGraph, beta::Float64; szverts=nothing)
  s = IndsNetwork(g; link_space=2)
  return ising_network(s, beta; szverts=szverts)
end

#Build an ITensor network on a graph specified by the inds network s. Bond_dim is given by link_space and entries are randomised (normal distribution, mean 0 std 1)
function randomITensorNetwork(s::IndsNetwork; link_space=nothing)
  ψ = ITensorNetwork(s; link_space)
  for v in vertices(ψ)
    ψᵥ = copy(ψ[v])
    randn!(ψᵥ)
    ψᵥ ./= sqrt(norm(ψᵥ))
    ψ[v] = ψᵥ
  end
  return ψ
end
