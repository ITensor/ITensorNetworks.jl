"""
ProjTTNOSum
"""
mutable struct ProjTTNOSum
  pm::Vector{ProjTTNO}
end

copy(P::ProjTTNOSum) = ProjTTNOSum(copy.(P.pm))

ProjTTNOSum(ttnos::Vector{TTNO}) = ProjTTNOSum([ProjTTNO(M) for M in ttnos])

ProjTTNOSum(Ms::TTNO...) = ProjTTNOSum([Ms...])

on_edge(P::ProjTTNOSum) = on_edge(P.pm[1])

nsite(P::ProjTTNOSum) = nsite(P.pm[1])

function set_nsite!(Ps::ProjTTNOSum, nsite)
  for P in Ps.pm
    set_nsite!(P, nsite)
  end
  return Ps
end

underlying_graph(P::ProjTTNOSum) = underlying_graph(P.pm[1])

Base.length(P::ProjTTNOSum) = length(P.pm[1])

sites(P::ProjTTNOSum) = sites(P.pm[1])

incident_edges(P::ProjTTNOSum) = incident_edges(P.pm[1])

internal_edges(P::ProjTTNOSum) = internal_edges(P.pm[1])

function product(P::ProjTTNOSum, v::ITensor)::ITensor
  Pv = product(P.pm[1], v)
  for n in 2:length(P.pm)
    Pv += product(P.pm[n], v)
  end
  return Pv
end

function Base.eltype(P::ProjTTNOSum)
  elT = eltype(P.pm[1])
  for n in 2:length(P.pm)
    elT = promote_type(elT, eltype(P.pm[n]))
  end
  return elT
end

(P::ProjTTNOSum)(v::ITensor) = product(P, v)

Base.size(P::ProjTTNOSum) = size(P.pm[1])

function position!(
  P::ProjTTNOSum, psi::TTNS, pos::Union{Vector{<:Tuple},NamedDimEdge{Tuple}}
)
  for M in P.pm
    position!(M, psi, pos)
  end
end
