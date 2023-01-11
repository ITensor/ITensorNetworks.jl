# make MPS behave like a tree without actually converting it

import Graphs: vertices, nv, ne, edgetype
import ITensors:
  AbstractProjMPO,
  orthocenter,
  position!,
  set_ortho_lims,
  tags,
  uniqueinds,
  siteinds,
  position!

const IsTreeState = Union{MPS,TTNS}
const IsTreeOperator = Union{MPO,TTNO}
const IsTreeProjOperator = Union{AbstractProjMPO,AbstractProjTTNO}
const IsTreeProjOperatorSum = Union{ProjMPOSum,ProjTTNOSum}

# number of vertices and edges
nv(psi::AbstractMPS) = length(psi)
ne(psi::AbstractMPS) = length(psi)-1

# support of effective hamiltonian
sites(P::AbstractProjMPO) = collect(ITensors.site_range(P))

# MPS lives on chain graph
underlying_graph(P::AbstractMPS) = chain_lattice_graph(length(P))
underlying_graph(P::AbstractProjMPO) = chain_lattice_graph(length(P.H))
underlying_graph(P::ProjMPOSum) = underlying_graph(P.pm[1])
vertices(psi::AbstractMPS) = vertices(underlying_graph(psi))

# default edgetype for ITensorNetworks
edgetype(::MPS) = NamedEdge{Int}

# catch-all constructors for projected operators
proj_operator(O::MPO) = ProjMPO(O)
proj_operator(O::TTNO) = ProjTTNO(O)
proj_operator_sum(Os::Vector{MPO}) = ProjMPOSum(Os)
proj_operator_sum(Os::Vector{<:TTNO}) = ProjTTNOSum(Os)
proj_operator_apply(psi0::MPS, O::MPO) = ProjMPOApply(psi0, O)
proj_operator_apply(psi0::TTNS, O::TTNO) = ProjTTNOApply(psi0, O)

# ortho lims as range versus ortho center as list of graph vertices
ortho_center(psi::MPS) = ortho_lims(psi)

function set_ortho_center(psi::MPS, oc::Vector)
  return set_ortho_lims(psi, first(oc):last(oc))
end

# setting number of sites of effective hamiltonian
set_nsite(P::AbstractProjMPO, nsite) = set_nsite!(copy(P), nsite)
set_nsite(P::ProjMPOSum, nsite) = set_nsite!(copy(P), nsite)

# setting position of effective hamiltonian on graph
position(P::AbstractProjMPO, psi::MPS, pos::Vector) = position!(copy(P), psi, minimum(pos))
position(P::AbstractProjMPO, psi::MPS, pos::NamedEdge) = position!(copy(P), psi, maximum(Tuple(pos)))
position(P::ProjMPOSum, args...) = position!(copy(P), args...)
# position!(::ProjMPOSum, ...) doesn't return the object; TODO: make this behave the same as position!() 
function position!(P::ProjMPOSum, psi::MPS, pos::Vector)
  position!(P, psi, minimum(pos))
  return P
end
function position(P::ProjMPOSum, psi::MPS, pos::NamedEdge)
  position!(P, psi, maximum(Tuple(pos)))
  return P
end

# link tags associated to a given graph edge
tags(psi::MPS, edge::NamedEdge) = tags(linkind(psi, minimum(Tuple(edge))))

# unique indices associated to the source of a graph edge
uniqueinds(psi::MPS, e::NamedEdge) = uniqueinds(psi[src(e)], psi[dst(e)])

# Observocalypse

const ObserverLike = Union{Observer,ITensors.AbstractObserver}

function obs_update!(observer::ObserverLike, psi::MPS, pos::Vector; kwargs...)
  bond = minimum(pos)
  return update!(observer; psi, bond, kwargs...)
end

function obs_update!(observer::ObserverLike, psi::MPS, pos::NamedEdge; kwargs...)
  return error("This should never be called!") # debugging...
end

function obs_update!(observer::ObserverLike, psi::TTNS, pos; kwargs...)
  return update!(observer; psi, pos, kwargs...)
end
