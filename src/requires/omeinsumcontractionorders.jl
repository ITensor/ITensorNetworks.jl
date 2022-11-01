# using OMEinsumContractionOrders
#using OMEinsumContractionOrders: CodeOptimizer

# Port OMEinsumContractionOrders to ITensors
# Slicing is not supported, because it might require extra work to slice an `ITensor` correctly.

const ITensorList = Union{Vector{<:ITensor},Tuple{Vararg{<:ITensor}}}

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
    args::Vector{Union{ITensorContractionTree, ITensor}}
    iy::Vector{Index{IT}}   # the output labels, note: this is type unstable
end
ITensors.inds(it::ITensorContractionTree) = (it.iy...,)

function ITensorContractionTree(args)::ITensorContractionTree
    args = Union{ITensorContractionTree, ITensor}[arg isa Union{AbstractVector, Tuple} ? ITensorContractionTree(arg) : arg for arg in args]
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

############################ Port to OMEinsumContractionOrders #######################
getid(index::Index) = index
getids(A::ITensor) = Index{Int}[getid(x) for x in ITensors.inds(A)]
getids(A::ITensorContractionTree) = collect(Index{Int}, getid.(ITensors.inds(A)))
function rootcode(it::ITensorContractionTree)
    ixs = [getids(A) for A in it.args]
    return OMEinsumContractionOrders.EinCode(ixs, it.iy)
end

function update_size_index_dict!(size_dict::Dict{Index{IT}}, index_dict::Dict{Index{IT}}, tensor::ITensor) where IT
    for ind in ITensors.inds(tensor)
        size_dict[getid(ind)] = ind.space
        index_dict[getid(ind)] = ind
    end
    return size_dict
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

# get a (labels, size_dict) representation of a ITensorContractionTree
function rawcode(tensors::ITensorList)
    # we use id as the label
    indsAs = [collect(Index{Int}, ITensors.inds(A)) for A in tensors]
    ixs = [getids(x) for x in tensors]
    unique_labels = unique(vcat(indsAs...))
    size_dict = Dict([getid(x)=>x.space for x in unique_labels])
    index_dict = Dict([getid(x)=>x for x in unique_labels])
    return OMEinsumContractionOrders.EinCode(ixs, getid.(infer_output(indsAs))), size_dict, index_dict
end

# infer the output tensor labels
function infer_output(inputs::AbstractVector{<:AbstractVector{Index{IT}}}) where IT
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

function rawcode(net::ITensorContractionTree{IT}) where IT
    size_dict = Dict{Index{IT},Int}()
    index_dict = Dict{Index{IT},Index{Int}}()
    r = rawcode!(net, size_dict, index_dict)
    return r, size_dict, index_dict
end
function rawcode!(net::ITensorContractionTree{IT}, size_dict::Dict{Index{IT}}, index_dict::Dict{Index{IT}}, index_counter=Base.RefValue(0)) where IT
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

"""
$(TYPEDSIGNATURES)
Optimize the contraction order of a tensor network specified as a vector tensors.
Returns a [`ITensorContractionTree`](@ref) instance, which can be evaluated with the [`evaluate`](@ref) function.
### Examples
```jldoctest
julia> using ITensors, ITensorContractionOrders
julia> i, j, k, l = Index(4), Index(5), Index(6), Index(7);
julia> x, y, z = randomITensor(i, j), randomITensor(j, k), randomITensor(k, l);
julia> net = optimize_contraction([x, y, z]; optimizer=TreeSA());
```
"""
function optimize_contraction(tensors::ITensorList; optimizer::OMEinsumContractionOrders.CodeOptimizer=TreeSA())
    r, size_dict, index_dict = rawcode(tensors)
    # merge vectors can speed up contraction order finding
    # optimize the permutation of tensors is set to true
    res = OMEinsumContractionOrders.optimize_code(r, size_dict, optimizer, OMEinsumContractionOrders.MergeVectors(), true)
    if res isa OMEinsumContractionOrders.SlicedEinsum   # slicing is not supported!
        if length(res.slicing) != 0
            @warn "Slicing is not yet supported by `ITensors`, removing slices..."
        end
        res = res.eins
    end
    return res
end

# decorate means converting the raw contraction pattern to ITensorContractionTree.
# `tensors` is the original input tensor list.
function convert_to_contraction_sequence(net::OMEinsumContractionOrders.NestedEinsum, tensor_indices)
  if OMEinsumContractionOrders.isleaf(net)
    return tensor_indices[net.tensorindex]
  else
    return convert_to_contraction_sequence.(net.args, Ref(tensor_indices))
  end
end

"""
Convert the result of `optimize_contraction` to a contraction sequence.
"""
function optimize_contraction_sequence(tensors::ITensorList; optimizer::OMEinsumContractionOrders.CodeOptimizer=TreeSA())
  res = optimize_contraction(tensors; optimizer)
  return convert_to_contraction_sequence(res, 1:length(tensors))
end

"""
$(TYPEDSIGNATURES)
"""
OMEinsumContractionOrders.peak_memory(net::ITensorContractionTree)::Int = peak_memory(rawcode(net)[1:2]...)

"""
$(TYPEDSIGNATURES)
"""
OMEinsumContractionOrders.flop(net::ITensorContractionTree)::Int = flop(rawcode(net)[1:2]...)

"""
$(TYPEDSIGNATURES)
"""
OMEinsumContractionOrders.timespacereadwrite_complexity(net::ITensorContractionTree) = OMEinsumContractionOrders.timespacereadwrite_complexity(rawcode(net)[1:2]...)

"""
$(TYPEDSIGNATURES)
"""
OMEinsumContractionOrders.timespace_complexity(net::ITensorContractionTree) = OMEinsumContractionOrders.timespacereadwrite_complexity(rawcode(net)[1:2]...)[1:2]

"""
$(TYPEDSIGNATURES)
"""
OMEinsumContractionOrders.contraction_complexity(net::ITensorContractionTree) = OMEinsumContractionOrders.contraction_complexity(rawcode(net)[1:2]...)

"""
$(TYPEDSIGNATURES)
    label_elimination_order(net::ITensorContractionTree) -> Vector
"""
function OMEinsumContractionOrders.label_elimination_order(net::ITensorContractionTree)
    r, size_dict, index_dict = rawcode(net)
    return getindex.(Ref(index_dict), label_elimination_order(r))
end

"""
    ITensorNetworks.contraction_sequence interface for OMEinsumContractionOrders

    GreedyMethod(; method=MinSpaceOut(), nrepeat=10)

The fast but poor greedy optimizer. Input arguments are:

* `method` is `MinSpaceDiff()` or `MinSpaceOut`.
    * `MinSpaceOut` choose one of the contraction that produces a minimum output tensor size,
    * `MinSpaceDiff` choose one of the contraction that decrease the space most.
* `nrepeat` is the number of repeatition, returns the best contraction order.


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
function contraction_sequence(alg::OMEinsumContractionOrders.CodeOptimizer, tensors::ITensorList)
  return optimize_contraction_sequence(tensors; optimizer=alg)
end
