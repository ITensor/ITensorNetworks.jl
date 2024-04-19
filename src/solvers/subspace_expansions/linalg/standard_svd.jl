#ToDo: pass use_*_cutoff as kwargs
function _svd_solve_normal(envMap, left_ind;
  maxdim,
  cutoff,
  use_relative_cutoff,
  use_absolute_cutoff)
  M = ITensors.ITensorNetworkMaps.contract(envMap)
  norm(M) ≤ eps(real(eltype(M))) && return nothing, nothing, nothing
  U, S, V = svd(
    M, left_ind; maxdim, cutoff, use_relative_cutoff, use_absolute_cutoff
  )
  vals = diag(array(S))
  (length(vals) == 1 && vals[1]^2 ≤ cutoff) && return nothing, nothing, nothing
  return U, S, V
end
