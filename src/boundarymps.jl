using ITensors: inner
using ITensors.ITensorMPS: ITensorMPS

#Given an ITensorNetwork on an Lx*Ly grid with sites indexed as (i,j) then perform contraction using a sequence of mps-mpo contractions
function contract_boundary_mps(tn::ITensorNetwork; kwargs...)
  dims = maximum(vertices(tn))
  d1, d2 = dims
  vL = ITensorMPS.MPS([tn[i1, 1] for i1 in 1:d1])
  for i2 in 2:(d2 - 2)
    T = ITensorMPS.MPO([tn[i1, i2] for i1 in 1:d1])
    vL = contract(T, vL; kwargs...)
  end
  T = ITensorMPS.MPO([tn[i1, d2 - 1] for i1 in 1:d1])
  vR = ITensorMPS.MPS([tn[i1, d2] for i1 in 1:d1])
  return inner(dag(vL), T, vR)[]
end
