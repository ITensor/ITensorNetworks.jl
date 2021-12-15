import Base: getindex, show, issubset

struct Sub{T}
  sub::T
end

get_sub(s::Sub) = s

struct SubIndex{S,I}
  sub::S
  index::I
end

function get_sub(si::SubIndex)
  if si.sub isa SubIndex
    return si.sub
  else
    return Sub(si.sub)
  end
end

SubIndex{S,I}(si::SubIndex{S,I}) where {S,I} = si

getindex(s::Sub, I...) = SubIndex(s.sub, I)
getindex(si::SubIndex, I...) = SubIndex(si, I)

issubset(s1::Sub, s2::Sub) = isequal(s1, s2)

function issubset(s1::SubIndex, s2::Union{Sub,SubIndex})
  isequal(s1, s2) && return true
  return issubset(get_sub(s1), s2)
end
issubset(s1::Sub, s2::SubIndex) = false

issubset(s2) = s1 -> issubset(s1, s2)

function show(io::IO, mime::MIME"text/plain", s::Sub)
  print(io, "Sub(")
  show(io, s.sub)
  print(io, ")")
  return nothing
end

show(io::IO, s::Sub) = show(io, MIME"text/plain"(), s)

function show(io::IO, mime::MIME"text/plain", si::SubIndex)
  show(io, get_sub(si))
  print(io, "[")
  show(io, only(si.index))
  print(io, "]")
  return nothing
end

show(io::IO, si::SubIndex) = show(io, MIME"text/plain"(), si)

