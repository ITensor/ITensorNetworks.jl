"""
ProjTTNSum
"""
struct ProjTTNSum{T<:AbstractProjTTN}
  pm::Vector{T} where {T}
  function ProjTTNSum(pm::Vector{T}) where {T}
    return new{T}(pm)
  end
end
#ToDo: define accessor functions

copy(P::ProjTTNSum) = ProjTTNSum(copy.(P.pm))

# The following constructors don't generalize well, maybe require to pass AbstractProjTTN instead of TTN and remove these? 
ProjTTNSum(ttnos::Vector{<:TTN}) = ProjTTNSum([ProjTTN(M) for M in ttnos])
function ProjTTNSum(init_states::Vector{<:TTN}, ttnos::Vector{<:TTN})
  return ProjTTNSum([
    ProjTTNApply(state, operator) for (state, operator) in zip(init_states, ttnos)
  ])
end

# This constructor can't differentiate between different concrete AbstractProjTTN, remove?
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

#ToDo remove parametrization? 
function position(P::ProjTTNSum, psi::TTN, pos)
  return ProjTTNSum(map(M -> position(M, psi, pos), P.pm))
end
