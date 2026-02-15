using Dictionaries: AbstractDictionary, Dictionary
using ITensors.NDTensors: NDTensors
using ITensors: ITensors, ITensor, Index, QN, filterinds, inds, op, replaceinds, uniqueinds
using NamedGraphs.Keys: Key

# Tensor sum: `A ⊞ B = A ⊗ Iᴮ + Iᴬ ⊗ B`
# https://github.com/JuliaLang/julia/issues/13333#issuecomment-143825995
# "PRESERVATION OF TENSOR SUM AND TENSOR PRODUCT"
# C. S. KUBRUSLY and N. LEVAN
# https://www.emis.de/journals/AMUC/_vol-80/_no_1/_kubrusly/kubrusly.pdf
function tensor_sum(A::ITensor, B::ITensor)
    extend_A = filterinds(uniqueinds(B, A); plev = 0)
    extend_B = filterinds(uniqueinds(A, B); plev = 0)
    for i in extend_A
        A *= op("I", i)
    end
    for i in extend_B
        B *= op("I", i)
    end
    return A + B
end

# Patch for contraction sequences with `Key`
# leaf values.
# TODO: Move patch to `ITensors.jl`.
ITensors._contract(As, index::Key) = As[index]

# TODO: Replace with a trait of the same name.
const IsIndexSpace = Union{<:Integer, Vector{<:Pair{QN, <:Integer}}}

# Infer the `Index` type of an `IndsNetwork` from the
# spaces that get input.
indtype(link_space::Nothing, site_space::Nothing) = Index
indtype(link_space::Nothing, site_space) = indtype(site_space)
indtype(link_space, site_space::Nothing) = indtype(link_space)
indtype(link_space, site_space) = promote_type(indtype(link_space), indtype(site_space))

# Default to type space
indtype(space) = _indtype(typeof(space))

# Base case
# Use `_indtype` to avoid recursion overflow
_indtype(T::Type{<:Index}) = T
_indtype(T::Type{<:IsIndexSpace}) = Index{T}
_indtype(::Type{Nothing}) = Index

# Containers
_indtype(T::Type{<:AbstractDictionary}) = _indtype(eltype(T))
_indtype(T::Type{<:AbstractVector}) = _indtype(eltype(T))

indtype(a::ITensor) = promote_indtype(typeof.(inds(a))...)

spacetype(::Index{T}) where {T} = T
spacetype(::Type{<:Index{T}}) where {T} = T

function promote_indtype(is::Vararg{Type{<:Index}})
    return reduce(promote_indtype_rule, is; init = Index{Int})
end

function promote_spacetype_rule(type1::Type, type2::Type)
    return error("Not implemented")
end

function promote_spacetype_rule(
        type1::Type{<:Integer}, type2::Type{<:Vector{<:Pair{QN, T2}}}
    ) where {T2 <: Integer}
    return Vector{Pair{QN, promote_type(type1, T2)}}
end

function promote_spacetype_rule(
        type1::Type{<:Vector{<:Pair{QN, <:Integer}}}, type2::Type{<:Integer}
    )
    return promote_spacetype_rule(type2, type1)
end

function promote_spacetype_rule(
        type1::Type{<:Vector{<:Pair{QN, T1}}}, type2::Type{<:Vector{<:Pair{QN, T2}}}
    ) where {T1 <: Integer, T2 <: Integer}
    return Vector{Pair{QN, promote_type(T1, T2)}}
end

function promote_spacetype_rule(type1::Type{<:Integer}, type2::Type{<:Integer})
    return promote_type(type1, type2)
end

function promote_indtype_rule(type1::Type{<:Index}, type2::Type{<:Index})
    return Index{promote_spacetype_rule(spacetype(type1), spacetype(type2))}
end

function promote_indtypeof end

trivial_space(x) = trivial_space(promote_indtypeof(x))
trivial_space(x::Type) = trivial_space(promote_indtype(x))

trivial_space(i::Type{<:Index{<:Integer}}) = 1
trivial_space(i::Type{<:Index{<:Vector{<:Pair{<:QN, <:Integer}}}}) = [QN() => 1]

"""
Given an input tensor and a Dict (ind_to_newind), replace inds of tensor that are also
keys of ind_to_newind to the value of ind_to_newind.
Note that it is the same as
ITensors.replaceinds(tensor, collect(keys(ind_to_newind)) => collect(values(ind_to_newind))).
Based on benchmark, this implementation is more efficient when the size of ind_to_newind is large.
TODO: we can remove this function once the original replaceinds performance is improved.
"""
function ITensors.replaceinds(tensor::ITensor, ind_to_newind::Dict{<:Index, <:Index})
    subset_inds = intersect(inds(tensor), collect(keys(ind_to_newind)))
    if length(subset_inds) == 0
        return tensor
    end
    out_inds = map(i -> ind_to_newind[i], subset_inds)
    return replaceinds(tensor, subset_inds => out_inds)
end

is_delta(it::ITensor) = is_delta(NDTensors.tensor(it))
is_delta(t::NDTensors.Tensor) = false
function is_delta(t::NDTensors.UniformDiagTensor)
    return isone(NDTensors.getdiagindex(t, 1))
end
