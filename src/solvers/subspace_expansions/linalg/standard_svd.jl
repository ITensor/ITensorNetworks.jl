#ToDo: pass use_*_cutoff as kwargs
function _svd_solve_normal(envMap, left_ind; maxdim, cutoff)
  M = ITensors.ITensorNetworkMaps.contract(envMap)
  # ToDo: infer eltype from envMap
  norm(M) ≤ eps(Float64) && return nothing, nothing, nothing
  U, S, V = svd(
    M, left_ind; maxdim, cutoff=cutoff, use_relative_cutoff=false, use_absolute_cutoff=true
  )
  vals = diag(array(S))
  (length(vals) == 1 && vals[1]^2 ≤ cutoff) && return nothing, nothing, nothing
  return U, S, V
end
