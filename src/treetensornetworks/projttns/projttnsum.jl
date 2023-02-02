"""
ProjTTNSum
"""
struct ProjTTNSum{V}
  pm::Vector{ProjTTN{V}}
  function ProjTTNSum(pm::Vector{ProjTTN{V}}) where {V}
    return new{V}(pm)
  end
end

copy(P::ProjTTNSum) = ProjTTNSum(copy.(P.pm))

ProjTTNSum(ttnos::Vector{<:TTN}) = ProjTTNSum([ProjTTN(M) for M in ttnos])

ProjTTNSum(Ms::TTN{V}...) where {V} = ProjTTNSum([Ms...])

on_edge(P::ProjTTNSum) = on_edge(P.pm[1])

nsite(P::ProjTTNSum) = nsite(P.pm[1])

function set_nsite(Ps::ProjTTNSum, nsite)
  return ProjTTNSum(map(M -> set_nsite(M, nsite), Ps.pm))
end

underlying_graph(P::ProjTTNSum) = underlying_graph(P.pm[1])

Base.length(P::ProjTTNSum) = length(P.pm[1])

sites(P::ProjTTNSum) = sites(P.pm[1])

incident_edges(P::ProjTTNSum) = incident_edges(P.pm[1])

internal_edges(P::ProjTTNSum) = internal_edges(P.pm[1])

function product(P::ProjTTNSum, v::ITensor)::ITensor
  Pv = product(P.pm[1], v)
  for n in 2:length(P.pm)
    Pv += product(P.pm[n], v)
  end
  return Pv
end

function Base.eltype(P::ProjTTNSum)
  elT = eltype(P.pm[1])
  for n in 2:length(P.pm)
    elT = promote_type(elT, eltype(P.pm[n]))
  end
  return elT
end

(P::ProjTTNSum)(v::ITensor) = product(P, v)

Base.size(P::ProjTTNSum) = size(P.pm[1])

function position(
  P::ProjTTNSum{V}, psi::TTN{V}, pos::Union{Vector{<:V},NamedEdge{V}}
) where {V}
  return ProjTTNSum(map(M -> position(M, psi, pos), P.pm))
end
