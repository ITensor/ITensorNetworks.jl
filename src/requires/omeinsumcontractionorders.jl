# OMEinsumContractionOrders wrapper for ITensors
# Slicing is not supported, because it might require extra work to slice an `ITensor` correctly.

const ITensorList = Union{Vector{<:ITensor},Tuple{Vararg{<:ITensor}}}

getid(index::Index) = index
getids(A::ITensor) = Index{Int}[getid(x) for x in ITensors.inds(A)]

# infer the output tensor labels
function infer_output(inputs::AbstractVector{<:AbstractVector{Index{IT}}}) where {IT}
  indslist = vcat(inputs...)
  # get output indices
  iy = Index{IT}[]
  for l in indslist
    c = count(==(l), indslist)
    if c == 1
      push!(iy, l)
    elseif c !== 2
      error("Each index in a tensor network must appear at most twice!")
    end
  end
  return iy
end

# get a (labels, size_dict) representation of a collection of ITensors
function rawcode(tensors::ITensorList)
  # we use id as the label
  indsAs = [collect(Index{Int}, ITensors.inds(A)) for A in tensors]
  ixs = [getids(x) for x in tensors]
  unique_labels = unique(vcat(indsAs...))
  size_dict = Dict([getid(x) => x.space for x in unique_labels])
  index_dict = Dict([getid(x) => x for x in unique_labels])
  return OMEinsumContractionOrders.EinCode(ixs, getid.(infer_output(indsAs))),
  size_dict,
  index_dict
end

"""
$(TYPEDSIGNATURES)
Optimize the contraction order of a tensor network specified as a vector tensors.
Returns a [`NestedEinsum`](@ref) instance.
### Examples
```jldoctest
julia> using ITensors, ITensorContractionOrders
julia> i, j, k, l = Index(4), Index(5), Index(6), Index(7);
julia> x, y, z = randomITensor(i, j), randomITensor(j, k), randomITensor(k, l);
julia> net = optimize_contraction([x, y, z]; optimizer=TreeSA());
```
"""
function optimize_contraction_nested_einsum(
  tensors::ITensorList; optimizer::OMEinsumContractionOrders.CodeOptimizer=TreeSA()
)
  r, size_dict, index_dict = rawcode(tensors)
  # merge vectors can speed up contraction order finding
  # optimize the permutation of tensors is set to true
  res = OMEinsumContractionOrders.optimize_code(
    r, size_dict, optimizer, OMEinsumContractionOrders.MergeVectors(), true
  )
  if res isa OMEinsumContractionOrders.SlicedEinsum   # slicing is not supported!
    if length(res.slicing) != 0
      @warn "Slicing is not yet supported by `ITensors`, removing slices..."
    end
    res = res.eins
  end
  return res
end

"""
Convert NestedEinsum to contraction sequence, such as `[[1, 2], [3, 4]]`.
"""
function convert_to_contraction_sequence(
  net::OMEinsumContractionOrders.NestedEinsum, tensor_indices
)
  if OMEinsumContractionOrders.isleaf(net)
    return tensor_indices[net.tensorindex]
  else
    return convert_to_contraction_sequence.(net.args, Ref(tensor_indices))
  end
end

"""
Convert the result of `optimize_contraction` to a contraction sequence.
"""
function optimize_contraction_sequence(
  tensors::ITensorList; optimizer::OMEinsumContractionOrders.CodeOptimizer=TreeSA()
)
  res = optimize_contraction_nested_einsum(tensors; optimizer)
  return convert_to_contraction_sequence(res, 1:length(tensors))
end
