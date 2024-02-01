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

ProjTTNSum(ttnos::Vector{<:TTN}) = ProjTTNSum([ProjTTN(M) for M in ttnos])

on_edge(P::ProjTTNSum) = on_edge(terms(P)[1])

nsite(P::ProjTTNSum) = nsite(terms(P)[1])

function set_nsite(Ps::ProjTTNSum, nsite)
  return ProjTTNSum(map(M -> set_nsite(M, nsite), Ps.terms))
end

underlying_graph(P::ProjTTNSum) = underlying_graph(terms(P)[1])

Base.length(P::ProjTTNSum) = length(terms(P)[1])

sites(P::ProjTTNSum) = sites(terms(P)[1])

incident_edges(P::ProjTTNSum) = incident_edges(terms(P)[1])

internal_edges(P::ProjTTNSum) = internal_edges(terms(P)[1])

function product(P::ProjTTNSum, v::ITensor)::ITensor
  Pv = product(terms(P)[1], v)
  for n in 2:length(terms(P))
    Pv += product(terms(P)[n], v)
  end
  return Pv
end

function contract(P::ProjTTNSum, v::ITensor)::ITensor
  Pv = contract(terms(P)[1], v)
  for n in 2:length(terms(P))
    Pv += contract(terms(P)[n], v)
  end
  return Pv
end

function Base.eltype(P::ProjTTNSum)
  elT = eltype(terms(P)[1])
  for n in 2:length(terms(P))
    elT = promote_type(elT, eltype(terms(P)[n]))
  end
  return elT
end

(P::ProjTTNSum)(v::ITensor) = product(P, v)

Base.size(P::ProjTTNSum) = size(terms(P)[1])

#ToDo remove parametrization? 
function position(P::ProjTTNSum, psi::TTN, pos)
  return ProjTTNSum(map(M -> position(M, psi, pos), terms(P)))
end
