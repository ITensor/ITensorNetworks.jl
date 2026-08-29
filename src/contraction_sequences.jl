using Graphs: nv, vertices
using ITensors.NDTensors: @Algorithm_str, Algorithm
using ITensors: ITensor
using NamedGraphs: NamedGraphs, decoded_vertex

# A key (index) type, used for unambiguously identifying an object as a key or
# index of an indexable object rather than as a container to descend into. That
# distinction matters for the nested structure of a contraction sequence, whose
# leaves are vertices that may themselves be containers:
#
#     [Key([1, 2]), [Key([3, 4]), Key([5, 6])]]
struct Key{K}
    I::K
end
Key(I...) = Key(I)

Base.show(io::IO, I::Key) = print(io, "Key(", I.I, ")")

NamedGraphs.to_graph_index(graph, key::Key) = key.I

const ITensorList = Union{Vector{ITensor}, Tuple{Vararg{ITensor}}}

function contraction_sequence(tn::ITensorList; alg = "optimal", kwargs...)
    return contraction_sequence(Algorithm(alg), tn; kwargs...)
end

function contraction_sequence(alg::Algorithm, tn::ITensorList)
    return throw(
        ArgumentError(
            "Algorithm $alg isn't defined for contraction sequence finding. Try loading a backend package like 
        TensorOperations.jl or OMEinsumContractionOrders.jl."
        )
    )
end

function deepmap(f, tree; filter = (x -> x isa AbstractArray))
    return filter(tree) ? map(t -> deepmap(f, t; filter = filter), tree) : f(tree)
end

function contraction_sequence(tn::AbstractITensorNetwork; kwargs...)
    ts = map(code -> tn[decoded_vertex(tn, code)], 1:nv(tn))
    seq_linear_index = contraction_sequence(ts; kwargs...)
    # TODO: Use `Functors.fmap` or `StructWalk`?
    return deepmap(code -> Key(decoded_vertex(tn, code)), seq_linear_index)
end
