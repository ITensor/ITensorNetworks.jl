#GIVEN AN ITENSOR NETWORK CORRESPONDING TO A Lx*Ly grid indexed as (i,j) then perform boundary sweeping using a max_dim of chi_max
function boundary_PEPS_contractor(
  psi::ITensorNetwork, Lx::Int64, Ly::Int64; chi_max=nothing, cut_off=nothing
)
  tn = deepcopy(psi)
  for i in 1:(Ly - 1)
    #First contract the row upwards
    for j in 1:Lx
      tn = ITensors.contract(tn, (i, j) => (i + 1, j))
    end
    #Now SVD all double Bonds on that row Down
    for j in 1:(Lx - 1)
      psi1 = tn[(i + 1, j)]
      psi2 = tn[(i + 1, j + 1)]

      A = psi1 * psi2
      inds_U = uniqueinds(A, NDTensors.inds(psi1))
      if (chi_max == nothing && cut_off != nothing)
        U, S, V = svd(A, inds_U; cutoff=cut_off)
      elseif (chi_max != nothing && cut_off == nothing)
        U, S, V = svd(A, inds_U; maxdim=chi_max)
      elseif (chi_max != nothing && cut_off != nothing)
        U, S, V = svd(A, inds_U; cutoff=cut_off, maxdim=chi_max)
      else
        U, S, V = svd(A, inds_U)
      end
      tn[(i + 1, j + 1)] = U * S
      tn[(i + 1, j)] = V
    end
  end

  Z = ITensors.contract(tn)

  return Z
end
