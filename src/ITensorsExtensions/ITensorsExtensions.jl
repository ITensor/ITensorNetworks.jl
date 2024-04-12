module ITensorsExtensions
using LinearAlgebra: LinearAlgebra, eigen, pinv
using ITensors:
  ITensor,
  Index,
  commonind,
  dag,
  hasqns,
  inds,
  isdiag,
  itensor,
  map_diag,
  noncommonind,
  noprime,
  replaceind,
  replaceinds,
  sim,
  space,
  sqrt_decomp
using ITensors.NDTensors:
  NDTensors,
  Block,
  Tensor,
  blockdim,
  blockoffsets,
  denseblocks,
  diaglength,
  getdiagindex,
  nzblocks,
  setdiagindex!,
  svd,
  tensor,
  DiagBlockSparseTensor,
  DenseTensor,
  BlockOffsets
using Observers: update!, insert_function!

function NDTensors.blockoffsets(dense::DenseTensor)
  return BlockOffsets{ndims(dense)}([Block(ntuple(Returns(1), ndims(dense)))], [0])
end
function NDTensors.nzblocks(dense::DenseTensor)
  return nzblocks(blockoffsets(dense))
end
NDTensors.blockdim(ind::Int, ::Block{1}) = ind
NDTensors.blockdim(i::Index{Int}, b::Integer) = blockdim(i, Block(b))
NDTensors.blockdim(i::Index{Int}, b::Block) = blockdim(space(i), b)

LinearAlgebra.isdiag(it::ITensor) = isdiag(tensor(it))

# Convenience functions
sqrt_diag(it::ITensor) = map_diag(sqrt, it)
inv_diag(it::ITensor) = map_diag(inv, it)
invsqrt_diag(it::ITensor) = map_diag(inv ∘ sqrt, it)
pinv_diag(it::ITensor) = map_diag(pinv, it)
pinvsqrt_diag(it::ITensor) = map_diag(pinv ∘ sqrt, it)

function map_itensor(
  f::Function, A::ITensor, linds=Index[first(inds(A))]; regularization=nothing, kwargs...
)
  rinds = setdiff(inds(A), linds)

  D, U = eigen(A, linds, rinds; kwargs...)
  ul, ur = noncommonind(D, U), commonind(D, U)
  ulnew = sim(noprime(ul))

  Ur = dag(U)
  Ul = replaceinds(U, (rinds..., ur), (linds..., ulnew))

  D_mapped = map_diag(s -> f(s + regularization), D)
  D_mapped = replaceind(D_mapped, ul, ulnew)

  sqrtD_mapped_L, δᵤᵥ, sqrtD_mapped_R = sqrt_decomp(D_mapped, ulnew, ur)
  sqrtD_mapped_R = denseblocks(sqrtD_mapped_R) * denseblocks(δᵤᵥ)

  return Ul * sqrtD_mapped_L * sqrtD_mapped_R * Ur
end

# Analagous to `denseblocks`.
# Extract the diagonal entries into a diagonal tensor.
function diagblocks(D::Tensor)
  nzblocksD = nzblocks(D)
  T = DiagBlockSparseTensor(eltype(D), nzblocksD, inds(D))
  for b in nzblocksD
    for n in 1:diaglength(D)
      setdiagindex!(T, getdiagindex(D, n), n)
    end
  end
  return T
end

diagblocks(it::ITensor) = itensor(diagblocks(tensor(it)))

"""Given a vector of ITensors, separate them into groups of commuting itensors (i.e. itensors in the same group do not share any common indices)"""
function group_commuting_itensors(its::Vector{ITensor})
  remaining_its = copy(its)
  it_groups = Vector{ITensor}[]

  while !isempty(remaining_its)
    cur_group = ITensor[]
    cur_indices = Index[]
    inds_to_remove = []
    for i in 1:length(remaining_its)
      it = remaining_its[i]
      it_inds = inds(it)

      if all([i ∉ cur_indices for i in it_inds])
        push!(cur_group, it)
        push!(cur_indices, it_inds...)
        push!(inds_to_remove, i)
      end
    end
    remaining_its = ITensor[
      remaining_its[i] for
      i in setdiff([i for i in 1:length(remaining_its)], inds_to_remove)
    ]
    push!(it_groups, cur_group)
  end

  return it_groups
end

end