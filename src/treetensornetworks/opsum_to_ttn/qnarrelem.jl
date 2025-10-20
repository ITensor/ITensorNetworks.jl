using StaticArrays: MVector

#####################################
# QNArrElem (sparse array with QNs) #
#####################################

struct QNArrElem{T, N}
    qn_idxs::MVector{N, QN}
    idxs::MVector{N, Int}
    val::T
end

function Base.:(==)(a1::QNArrElem{T, N}, a2::QNArrElem{T, N})::Bool where {T, N}
    return (a1.idxs == a2.idxs && a1.val == a2.val && a1.qn_idxs == a2.qn_idxs)
end

function Base.isless(a1::QNArrElem{T, N}, a2::QNArrElem{T, N})::Bool where {T, N}
    ###two separate loops s.t. the MPS case reduces to the ITensors Implementation of QNMatElem
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
