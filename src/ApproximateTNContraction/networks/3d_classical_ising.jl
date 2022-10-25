using ITensors

function ising_mpo(
  pairs::Vector{<:Pair{<:Index,<:Index}}, β::Real, J::Real=1.0; sz::Bool=false
)
  d = dim(pairs[1].first)
  for p in pairs
    @assert d == dim(p.first) == dim(p.second)
  end
  indices = mapreduce(p -> [p.first, p.second], vcat, pairs)
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
  npairs = floor(Int, length(inds) / 2)
  @assert length(inds) == 2 * npairs
  βc = 0.5 * log(√2 + 1)
  β = 1.0 * βc
  pairs = [Pair(inds[2 * i - 1], inds[2 * i]) for i in 1:npairs]
  return ising_mpo(pairs, β)
end

function ising_partition(N, d=2)
  tn_inds = inds_network(N...; linkdims=d, periodic=true)
  tn = map(inds -> isingTensor(inds), tn_inds)
  return project_boundary(tn)
end
