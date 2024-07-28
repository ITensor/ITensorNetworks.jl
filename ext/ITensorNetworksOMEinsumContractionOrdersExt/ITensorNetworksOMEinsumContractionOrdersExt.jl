module ITensorNetworksOMEinsumContractionOrdersExt
using DocStringExtensions: TYPEDSIGNATURES
using ITensorNetworks: ITensorNetworks
using ITensors: ITensors, Index, ITensor, inds
using NDTensors: dim
using NDTensors.AlgorithmSelection: @Algorithm_str
using OMEinsumContractionOrders: OMEinsumContractionOrders

# OMEinsumContractionOrders wrapper for ITensors
# Slicing is not supported, because it might require extra work to slice an `ITensor` correctly.

const ITensorList = Union{Vector{ITensor},Tuple{Vararg{ITensor}}}

# infer the output tensor labels
# TODO: Use `symdiff` instead.
function infer_output(inputs::AbstractVector{<:AbstractVector{<:Index}})
  indslist = reduce(vcat, inputs)
  # get output indices
  iy = eltype(eltype(inputs))[]
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
  ixs = collect.(inds.(tensors))
  unique_labels = unique(reduce(vcat, indsAs))
  size_dict = Dict([x => dim(x) for x in unique_labels])
  index_dict = Dict([x => x for x in unique_labels])
  return OMEinsumContractionOrders.EinCode(ixs, infer_output(indsAs)), size_dict, index_dict
end

"""
$(TYPEDSIGNATURES)
Optimize the contraction order of a tensor network specified as a vector tensors.
Returns a [`NestedEinsum`](@ref) instance.
### Examples
```jldoctest
julia> using ITensors, ITensorContractionOrders
julia> i, j, k, l = Index(4), Index(5), Index(6), Index(7);
julia> x, y, z = random_itensor(i, j), random_itensor(j, k), random_itensor(k, l);
julia> net = optimize_contraction([x, y, z]; optimizer=TreeSA());
```
"""
function optimize_contraction_nested_einsum(
  tensors::ITensorList;
  optimizer::OMEinsumContractionOrders.CodeOptimizer=OMEinsumContractionOrders.TreeSA(),
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
function convert_to_contraction_sequence(net::OMEinsumContractionOrders.NestedEinsum)
  if OMEinsumContractionOrders.isleaf(net)
    return net.tensorindex
  else
    return convert_to_contraction_sequence.(net.args)
  end
end

"""
Convert the result of `optimize_contraction` to a contraction sequence.
"""
function optimize_contraction_sequence(
  tensors::ITensorList; optimizer::OMEinsumContractionOrders.CodeOptimizer=TreeSA()
)
  res = optimize_contraction_nested_einsum(tensors; optimizer)
  return convert_to_contraction_sequence(res)
end

"""
    GreedyMethod(; method=MinSpaceOut(), nrepeat=10)

The fast but poor greedy optimizer. Input arguments are:

* `method` is `MinSpaceDiff()` or `MinSpaceOut`.
    * `MinSpaceOut` choose one of the contraction that produces a minimum output tensor size,
    * `MinSpaceDiff` choose one of the contraction that decrease the space most.
* `nrepeat` is the number of repeatition, returns the best contraction order.
"""
function ITensorNetworks.contraction_sequence(
  ::Algorithm"greedy", tn::Vector{ITensor}; kwargs...
)
  return optimize_contraction_sequence(
    tn; optimizer=OMEinsumContractionOrders.GreedyMethod(; kwargs...)
  )
end

"""
    TreeSA(; sc_target=20, βs=collect(0.01:0.05:15), ntrials=10, niters=50,
             sc_weight=1.0, rw_weight=0.2, initializer=:greedy, greedy_config=GreedyMethod(; nrepeat=1))

Optimize the einsum contraction pattern using the simulated annealing on tensor expression tree.

* `sc_target` is the target space complexity,
* `ntrials`, `βs` and `niters` are annealing parameters, doing `ntrials` indepedent annealings, each has inverse tempteratures specified by `βs`, in each temperature, do `niters` updates of the tree.
* `sc_weight` is the relative importance factor of space complexity in the loss compared with the time complexity.
* `rw_weight` is the relative importance factor of memory read and write in the loss compared with the time complexity.
* `initializer` specifies how to determine the initial configuration, it can be `:greedy` or `:random`. If it is using `:greedy` method to generate the initial configuration, it also uses two extra arguments `greedy_method` and `greedy_nrepeat`.
* `nslices` is the number of sliced legs, default is 0.
* `fixed_slices` is a vector of sliced legs, default is `[]`.

### References
* [Recursive Multi-Tensor Contraction for XEB Verification of Quantum Circuits](https://arxiv.org/abs/2108.05665)
"""
function ITensorNetworks.contraction_sequence(::Algorithm"tree_sa", tn; kwargs...)
  return optimize_contraction_sequence(
    tn; optimizer=OMEinsumContractionOrders.TreeSA(; kwargs...)
  )
end

"""
    SABipartite(; sc_target=25, ntrials=50, βs=0.1:0.2:15.0, niters=1000
                  max_group_size=40, greedy_config=GreedyMethod(), initializer=:random)

Optimize the einsum code contraction order using the Simulated Annealing bipartition + Greedy approach.
This program first recursively cuts the tensors into several groups using simulated annealing,
with maximum group size specifed by `max_group_size` and maximum space complexity specified by `sc_target`,
Then finds the contraction order inside each group with the greedy search algorithm. Other arguments are:

* `size_dict`, a dictionary that specifies leg dimensions,
* `sc_target` is the target space complexity, defined as `log2(number of elements in the largest tensor)`,
* `max_group_size` is the maximum size that allowed to used greedy search,
* `βs` is a list of inverse temperature `1/T`,
* `niters` is the number of iteration in each temperature,
* `ntrials` is the number of repetition (with different random seeds),
* `greedy_config` configures the greedy method,
* `initializer`, the partition configuration initializer, one can choose `:random` or `:greedy` (slow but better).

### References
* [Hyper-optimized tensor network contraction](https://arxiv.org/abs/2002.01935)
"""
function ITensorNetworks.contraction_sequence(::Algorithm"sa_bipartite", tn; kwargs...)
  return optimize_contraction_sequence(
    tn; optimizer=OMEinsumContractionOrders.SABipartite(; kwargs...)
  )
end

"""
    KaHyParBipartite(; sc_target, imbalances=collect(0.0:0.005:0.8),
                       max_group_size=40, greedy_config=GreedyMethod())

Optimize the einsum code contraction order using the KaHyPar + Greedy approach.
This program first recursively cuts the tensors into several groups using KaHyPar,
with maximum group size specifed by `max_group_size` and maximum space complexity specified by `sc_target`,
Then finds the contraction order inside each group with the greedy search algorithm. Other arguments are:

* `sc_target` is the target space complexity, defined as `log2(number of elements in the largest tensor)`,
* `imbalances` is a KaHyPar parameter that controls the group sizes in hierarchical bipartition,
* `max_group_size` is the maximum size that allowed to used greedy search,
* `greedy_config` is a greedy optimizer.

### References
* [Hyper-optimized tensor network contraction](https://arxiv.org/abs/2002.01935)
* [Simulating the Sycamore quantum supremacy circuits](https://arxiv.org/abs/2103.03074)
"""
function ITensorNetworks.contraction_sequence(::Algorithm"kahypar_bipartite", tn; kwargs...)
  return optimize_contraction_sequence(
    tn; optimizer=OMEinsumContractionOrders.KaHyParBipartite(; kwargs...)
  )
end
end
