"""
ProjTTNSum
"""
struct ProjTTNSum{T<:AbstractProjTTN}
  terms::Vector{T}

  function ProjTTNSum(terms::Vector{<:AbstractProjTTN})
    return new{eltype(terms)}(terms)
  end
end

terms(P::ProjTTNSum) = P.terms

copy(P::ProjTTNSum) = ProjTTNSum(copy.(terms(P)))

ProjTTNSum(operators::Vector{<:AbstractTTN}) = ProjTTNSum(ProjTTN.(operators))

on_edge(P::ProjTTNSum) = on_edge(terms(P)[1])

nsite(P::ProjTTNSum) = nsite(terms(P)[1])

function set_nsite(Ps::ProjTTNSum, nsite)
  return ProjTTNSum(map(p -> set_nsite(p, nsite), terms(Ps)))
end

underlying_graph(P::ProjTTNSum) = underlying_graph(terms(P)[1])

Base.length(P::ProjTTNSum) = length(terms(P)[1])

sites(P::ProjTTNSum) = sites(terms(P)[1])

incident_edges(P::ProjTTNSum) = incident_edges(terms(P)[1])

internal_edges(P::ProjTTNSum) = internal_edges(terms(P)[1])

product(P::ProjTTNSum, v::ITensor) = noprime(contract(P, v))

contract(P::ProjTTNSum, v::ITensor)::ITensor = sum(p -> contract(p, v), terms(P))

function Base.eltype(P::ProjTTNSum)
  return mapreduce(eltype, promote_type, terms(P))
end

(P::ProjTTNSum)(v::ITensor) = product(P, v)

Base.size(P::ProjTTNSum) = size(terms(P)[1])

function position(P::ProjTTNSum, psi::AbstractTTN, pos)
  return ProjTTNSum(map(M -> position(M, psi, pos), terms(P)))
end
