
#####################################
# QNArrElem (sparse array with QNs) #
#####################################

struct QNArrElem{T}
  qn_idxs::Vector{QN}
  idxs::Vector{Int}
  val::T
end

function Base.:(==)(a1::QNArrElem{T}, a2::QNArrElem{T})::Bool where {T}
  return (a1.idxs == a2.idxs && a1.val == a2.val && a1.qn_idxs == a2.qn_idxs)
end

function Base.isless(a1::QNArrElem{T}, a2::QNArrElem{T})::Bool where {T}
  ###two separate loops s.t. the MPS case reduces to the ITensors Implementation of QNMatElem
  N = length(a1.qn_idxs)
  @assert N == length(a2.qn_idxs)
  for n in 1:N
    if a1.qn_idxs[n] != a2.qn_idxs[n]
      return a1.qn_idxs[n] < a2.qn_idxs[n]
    end
  end
  for n in 1:N
    if a1.idxs[n] != a2.idxs[n]
      return a1.idxs[n] < a2.idxs[n]
    end
  end
  return a1.val < a2.val
end
