using ITensorMPS: ITensorMPS

function ITensorMPS.MPO(opsum::OpSum, s::IndsNetwork)
  s_linear = [only(s[v]) for v in 1:nv(s)]
  return ITensorMPS.MPO(opsum, s_linear)
end

function ITensorMPS.MPO(opsum_sum::Sum{<:OpSum}, s::IndsNetwork)
  return ITensorMPS.MPO(sum(Ops.terms(opsum_sum)), s)
end

function ITensorMPS.random_mps(s::IndsNetwork, args...; kwargs...)
  s_linear = [only(s[v]) for v in 1:nv(s)]
  return ITensorMPS.random_mps(s_linear, args...; kwargs...)
end

function ITensorMPS.MPS(s::IndsNetwork, args...; kwargs...)
  s_linear = [only(s[v]) for v in 1:nv(s)]
  return ITensorMPS.MPS(s_linear, args...; kwargs...)
end
