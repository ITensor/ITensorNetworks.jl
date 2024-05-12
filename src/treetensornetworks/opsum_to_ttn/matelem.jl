
struct MatElem{T}
  row::Int
  col::Int
  val::T
end

Base.eltype(::MatElem{T}) where {T} = T

function col_major_order(el1::MatElem, el2::MatElem)
  if el1.col == el2.col
    return el1.row < el2.row
  end
  return el1.col < el2.col
end

function to_matrix(els::Vector{<:MatElem})
  els = sort(els; lt=col_major_order)
  nr = maximum(el -> el.row, els)
  nc = last(els).col
  M = zeros(eltype(first(els)), nr, nc)
  for el in els
    M[el.row, el.col] = el.val
  end
  return M
end
