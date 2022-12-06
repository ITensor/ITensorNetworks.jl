using ITensors

function ising_mpo(indices::Vector, β::Real, J::Real=1.0; sz::Bool=false)
  d = dim(indices[1])
  for ind in indices
    @assert d == dim(ind)
  end
  order = length(indices)
  T = ITensor(indices...)
  for i in 1:d
    index = [i for _ in 1:order]
    T[index...] = 1.0
  end
  if sz
    index = [1 for _ in 1:order]
    T[index...] = -1.0
  end
  simindices = map(sim, indices)
  for i in 1:length(indices)
    T = T * delta(indices[i], simindices[i])
  end

  f(λ₊, λ₋) = [
    (λ₊ + λ₋)/2 (λ₊ - λ₋)/2
    (λ₊ - λ₋)/2 (λ₊ + λ₋)/2
  ]
  λ₊ = √(exp(β * J) + exp(-β * J))
  λ₋ = √(exp(β * J) - exp(-β * J))
  X = f(λ₊, λ₋)

  for i in 1:length(indices)
    Xh = itensor(vec(X), simindices[i], indices[i])
    T = T * Xh
  end
  return T
end

function isingTensor(inds::Vector)
  # βc = 0.5 * log(√2 + 1)
  β = 0.3 # 1.0 * βc
  return ising_mpo(inds, β)
end

function ising_partition(N, d=2)
  tn_inds = inds_network(N...; linkdims=d, periodic=false)
  return map(inds -> isingTensor(inds), tn_inds)
end
