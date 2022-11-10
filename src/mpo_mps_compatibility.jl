function ITensors.MPO(opsum::OpSum, s::IndsNetwork)
  s_linear = [only(s[v]) for v in 1:nv(s)]
  return MPO(opsum, s_linear)
end

function ITensors.MPO(opsum_sum::Sum{<:OpSum}, s::IndsNetwork)
  return MPO(sum(Ops.terms(opsum_sum)), s)
end

function ITensors.randomMPS(s::IndsNetwork, args...; kwargs...)
  s_linear = [only(s[v]) for v in 1:nv(s)]
  return randomMPS(s_linear, args...; kwargs...)
end

function ITensors.MPS(s::IndsNetwork, args...; kwargs...)
  s_linear = [only(s[v]) for v in 1:nv(s)]
  return MPS(s_linear, args...; kwargs...)
end
