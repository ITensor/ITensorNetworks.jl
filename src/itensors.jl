# Tensor sum: `A ⊞ B = A ⊗ Iᴮ + Iᴬ ⊗ B`
# https://github.com/JuliaLang/julia/issues/13333#issuecomment-143825995
# "PRESERVATION OF TENSOR SUM AND TENSOR PRODUCT"
# C. S. KUBRUSLY and N. LEVAN
# https://www.emis.de/journals/AMUC/_vol-80/_no_1/_kubrusly/kubrusly.pdf
function tensor_sum(A::ITensor, B::ITensor)
  extend_A = filterinds(uniqueinds(B, A); plev=0)
  extend_B = filterinds(uniqueinds(A, B); plev=0)
  for i in extend_A
    A *= op("I", i)
  end
  for i in extend_B
    B *= op("I", i)
  end
  return A + B
end

# TODO: Replace with a trait?
const ITensorCollection = Union{Vector{ITensor},Dictionary{<:Any,ITensor}}

# Patch for contraction sequences with `Key`
# leaf values.
# TODO: Move patch to `ITensors.jl`.
ITensors._contract(As, index::Key) = As[index]

spacetype(::Type{Index}) = Int
spacetype(::Type{<:Index{T}}) where {T} = T
spacetype(T::Type{<:Vector}) = spacetype(eltype(T))

trivial_space(::Type{<:Integer}) = 1
trivial_space(::Type{<:Pair{QN}}) = (QN() => 1)
trivial_space(T::Type{<:Vector{<:Pair{QN}}}) = [trivial_space(eltype(T))]

_trivial_space(T::Type) = trivial_space(spacetype(T))
_trivial_space(x::Any) = trivial_space(typeof(x))

trivial_space(T::Type{<:Index}) = _trivial_space(T)
trivial_space(T::Type{<:Vector}) = _trivial_space(T)

trivial_space(x::Index) = _trivial_space(x)
trivial_space(x::Vector{<:Index}) = _trivial_space(x)
trivial_space(x::ITensor) = trivial_space(inds(x))
trivial_space(x::Tuple{Vararg{Index}}) = trivial_space(first(x))
