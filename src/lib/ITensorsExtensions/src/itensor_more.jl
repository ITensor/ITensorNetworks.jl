using Dictionaries: AbstractDictionary
using ITensors: ITensor, Index, QN, filterinds, op, uniqueinds

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

# Spaces accepted by `Index`: a plain `Integer` dimension (dense) or a QN-block
# vector `Vector{Pair{QN, <:Integer}}` (block-sparse). Used as the dispatch tag
# for `_indtype` so a raw space spec gets mapped to the appropriate `Index{T}`.
const IsIndexSpace = Union{<:Integer, Vector{<:Pair{QN, <:Integer}}}

# Infer the `Index` type of an `IndsNetwork` from the per-edge / per-vertex
# space specs the user passes to the `IndsNetwork{V, I}(g, link_space,
# site_space)` constructor. Either argument may be `nothing` (defer entirely
# to the other), a space spec, or a container (`AbstractVector` /
# `AbstractDictionary`) of space specs.
indtype(link_space::Nothing, site_space::Nothing) = Index
indtype(link_space::Nothing, site_space) = indtype(site_space)
indtype(link_space, site_space::Nothing) = indtype(link_space)
indtype(link_space, site_space) = promote_type(indtype(link_space), indtype(site_space))

indtype(space) = _indtype(typeof(space))

# `_indtype` dispatches on the static space type to avoid recursion through
# the user-facing `indtype(space)` entry point.
_indtype(T::Type{<:Index}) = T
_indtype(T::Type{<:IsIndexSpace}) = Index{T}
_indtype(::Type{Nothing}) = Index
_indtype(T::Type{<:AbstractDictionary}) = _indtype(eltype(T))
_indtype(T::Type{<:AbstractVector}) = _indtype(eltype(T))
