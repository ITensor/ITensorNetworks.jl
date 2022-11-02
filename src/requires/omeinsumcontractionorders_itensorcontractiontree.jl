"""
$(TYPEDEF)
    ITensorContractionTree(args) -> ITensorContractionTree
Define a tensor network with its contraction order specified by a tree structure.
In this network, each index in this tensor network must appear either twice or once.
The input `args` is a Vector of [`ITensor`](@ref) or another layer of Vector.
This data type can be automatically generated from [`optimize_contraction`](@ref) function.
### Fields
$(TYPEDFIELDS)
### Examples
The following code creates a tensor network and evaluates it in a sequencial order.
```jldoctest
julia> using ITensors, ITensorContractionOrders
julia> i, j, k, l = Index(4), Index(5), Index(6), Index(7);
julia> x, y, z = randomITensor(i, j), randomITensor(j, k), randomITensor(k, l);
julia> it = ITensorContractionTree([[x, y] ,z]);
julia> itensor_list = ITensorContractionOrders.flatten(it);  # convert this tensor network to a Vector of ITensors
julia> evaluate(it) ≈ foldl(*, itensor_list)
true
```
"""
struct ITensorContractionTree{IT}
  args::Vector{Union{ITensorContractionTree,ITensor}}
  iy::Vector{Index{IT}}   # the output labels, note: this is type unstable
end
ITensors.inds(it::ITensorContractionTree) = (it.iy...,)

function ITensorContractionTree(args)::ITensorContractionTree
  args = Union{ITensorContractionTree,ITensor}[
    arg isa Union{AbstractVector,Tuple} ? ITensorContractionTree(arg) : arg for arg in args
  ]
  # get output labels
  # NOTE: here we assume the output index id has `Int` type
  labels = collect.(Index{Int}, ITensors.inds.(args))
  return ITensorContractionTree(args, infer_output(labels))
end

"""
    flatten(it::ITensorContractionTree) -> Vector
Convert an [`ITensorContractionTree`](@ref) to a Vector of [`ITensor`](@ref).
"""
flatten(it::ITensorContractionTree) = flatten!(it, ITensor[])
function flatten!(it::ITensorContractionTree, lst)
  for arg in it.args
    if arg isa ITensor
      push!(lst, arg)
    else
      flatten!(arg, lst)
    end
  end
  return lst
end

# Contract and evaluate an itensor network.
"""
$(TYPEDSIGNATURES)
"""
evaluate(it::ITensorContractionTree)::ITensor = foldl(*, evaluate.(it.args))
evaluate(it::ITensor) = it

getids(A::ITensorContractionTree) = collect(Index{Int}, getid.(ITensors.inds(A)))
function rootcode(it::ITensorContractionTree)
  ixs = [getids(A) for A in it.args]
  return OMEinsumContractionOrders.EinCode(ixs, it.iy)
end

# decorate means converting the raw contraction pattern to ITensorContractionTree.
# `tensors` is the original input tensor list.
function decorate(net::OMEinsumContractionOrders.NestedEinsum, tensors::ITensorList)
  if OMEinsumContractionOrders.isleaf(net)
    return tensors[net.tensorindex]
  else
    return ITensorContractionTree(decorate.(net.args, Ref(tensors)))
  end
end

function update_size_index_dict!(
  size_dict::Dict{Index{IT}}, index_dict::Dict{Index{IT}}, tensor::ITensor
) where {IT}
  for ind in ITensors.inds(tensor)
    size_dict[getid(ind)] = ind.space
    index_dict[getid(ind)] = ind
  end
  return size_dict
end

function rawcode!(
  net::ITensorContractionTree{IT},
  size_dict::Dict{Index{IT}},
  index_dict::Dict{Index{IT}},
  index_counter=Base.RefValue(0),
) where {IT}
  args = map(net.args) do s
    if s isa ITensor
      update_size_index_dict!(size_dict, index_dict, s)
      OMEinsumContractionOrders.NestedEinsum{Index{IT}}(index_counter[] += 1)
    else  # ITensorContractionTree
      scode = rawcode!(s, size_dict, index_dict, index_counter)
      # no need to update size, size is only updated on the leaves.
      scode
    end
  end
  return OMEinsumContractionOrders.NestedEinsum(args, rootcode(net))
end
function rawcode(net::ITensorContractionTree{IT}) where {IT}
  size_dict = Dict{Index{IT},Int}()
  index_dict = Dict{Index{IT},Index{Int}}()
  r = rawcode!(net, size_dict, index_dict)
  return r, size_dict, index_dict
end

"""
$(TYPEDSIGNATURES)
"""
OMEinsumContractionOrders.peak_memory(net::ITensorContractionTree)::Int =
  peak_memory(rawcode(net)[1:2]...)

"""
$(TYPEDSIGNATURES)
"""
OMEinsumContractionOrders.flop(net::ITensorContractionTree)::Int =
  flop(rawcode(net)[1:2]...)

"""
$(TYPEDSIGNATURES)
"""
function OMEinsumContractionOrders.timespacereadwrite_complexity(
  net::ITensorContractionTree
)
  return OMEinsumContractionOrders.timespacereadwrite_complexity(rawcode(net)[1:2]...)
end

"""
$(TYPEDSIGNATURES)
"""
function OMEinsumContractionOrders.timespace_complexity(net::ITensorContractionTree)
  return OMEinsumContractionOrders.timespacereadwrite_complexity(rawcode(net)[1:2]...)[1:2]
end

"""
$(TYPEDSIGNATURES)
"""
function OMEinsumContractionOrders.contraction_complexity(net::ITensorContractionTree)
  return OMEinsumContractionOrders.contraction_complexity(rawcode(net)[1:2]...)
end

"""
$(TYPEDSIGNATURES)
    label_elimination_order(net::ITensorContractionTree) -> Vector
"""
function OMEinsumContractionOrders.label_elimination_order(net::ITensorContractionTree)
  r, size_dict, index_dict = rawcode(net)
  return getindex.(Ref(index_dict), label_elimination_order(r))
end
