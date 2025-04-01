using LinearAlgebra: LinearAlgebra, eigen, pinv
using ITensorBase: ITensor
## using ITensorBase:
##   ITensor,
##   Index,
##   commonind,
##   dag,
##   hasqns,
##   inds,
##   isdiag,
##   itensor,
##   map_diag,
##   noncommonind,
##   noprime,
##   replaceind,
##   replaceinds,
##   sim,
##   space,
##   sqrt_decomp
## using ITensors.NDTensors:
##   NDTensors,
##   Block,
##   Tensor,
##   blockdim,
##   blockoffsets,
##   denseblocks,
##   diaglength,
##   getdiagindex,
##   nzblocks,
##   setdiagindex!,
##   svd,
##   tensor,
##   DiagBlockSparseTensor,
##   DenseTensor,
##   BlockOffsets

## function NDTensors.blockoffsets(dense::DenseTensor)
##   return BlockOffsets{ndims(dense)}([Block(ntuple(Returns(1), ndims(dense)))], [0])
## end
## function NDTensors.nzblocks(dense::DenseTensor)
##   return nzblocks(blockoffsets(dense))
## end
## NDTensors.blockdim(ind::Int, ::Block{1}) = ind
## NDTensors.blockdim(i::Index{Int}, b::Integer) = blockdim(i, Block(b))
## NDTensors.blockdim(i::Index{Int}, b::Block) = blockdim(space(i), b)

## LinearAlgebra.isdiag(it::ITensor) = isdiag(tensor(it))

# Convenience functions
sqrt_diag(it::ITensor) = map_diag(sqrt, it)
inv_diag(it::ITensor) = map_diag(inv, it)
invsqrt_diag(it::ITensor) = map_diag(inv ∘ sqrt, it)
pinv_diag(it::ITensor) = map_diag(pinv, it)
pinvsqrt_diag(it::ITensor) = map_diag(pinv ∘ sqrt, it)

#TODO: Make this work for non-hermitian A
function eigendecomp(A::ITensor, linds, rinds; ishermitian=false, kwargs...)
  @assert ishermitian
  D, U = eigen(A, linds, rinds; ishermitian, kwargs...)
  ul, ur = noncommonind(D, U), commonind(D, U)
  Ul = replaceinds(U, vcat(rinds, ur), vcat(linds, ul))

  return Ul, D, dag(U)
end

function map_eigvals(f::Function, A::ITensor, inds...; ishermitian=false, kwargs...)
  if isdiag(A)
    return map_diag(f, A)
  end

  Ul, D, Ur = eigendecomp(A, inds...; ishermitian, kwargs...)

  return Ul * map_diag(f, D) * Ur
end

## # Analagous to `denseblocks`.
## # Extract the diagonal entries into a diagonal tensor.
## function diagblocks(D::Tensor)
##   nzblocksD = nzblocks(D)
##   T = DiagBlockSparseTensor(eltype(D), nzblocksD, inds(D))
##   for b in nzblocksD
##     for n in 1:diaglength(D)
##       setdiagindex!(T, getdiagindex(D, n), n)
##     end
##   end
##   return T
## end

## diagblocks(it::ITensor) = itensor(diagblocks(tensor(it)))
