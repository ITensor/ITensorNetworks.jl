using ITensors: ITensor, commonind, dag, isdiag, map_diag, noncommonind, replaceinds
using LinearAlgebra: eigen

#TODO: Make this work for non-hermitian A
function eigendecomp(A::ITensor, linds, rinds; ishermitian = false, kwargs...)
    @assert ishermitian
    D, U = eigen(A, linds, rinds; ishermitian, kwargs...)
    ul, ur = noncommonind(D, U), commonind(D, U)
    Ul = replaceinds(U, vcat(rinds, ur), vcat(linds, ul))
    return Ul, D, dag(U)
end

function map_eigvals(f::Function, A::ITensor, Linds, Rinds; kws...)
    isdiag(A) && return map_diag(f, A)
    Ul, D, Ur = eigendecomp(A, Linds, Rinds; kws...)
    return Ul * map_diag(f, D) * Ur
end
