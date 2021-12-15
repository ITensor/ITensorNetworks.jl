#
# SubIndexing
#

using Dictionaries

import Base: getindex

struct Sub{T}
  sub::T
end

struct SubIndex{S,I}
  sub::S
  index::I
end

getindex(s::Sub, I...) = SubIndex(s, I)
getindex(s::SubIndex, I...) = SubIndex(s, I)

_getindex(x, SI::SubIndex) = x[SI.sub][SI.index...]
_getindex(x, S::Sub) = x[S.sub]
getindex(A::AbstractArray, SI::Union{Sub,SubIndex}) = _getindex(A, SI)
getindex(A::AbstractDict, SI::Union{Sub,SubIndex}) = _getindex(A, SI)
getindex(A::Dict, SI::Union{Sub,SubIndex}) = _getindex(A, SI)
getindex(A::AbstractDictionary, SI::Union{Sub,SubIndex}) = _getindex(A, SI)
