#
# HyperCubic lattice
#

# An N-dimensional hypercubic lattice with periodic
# boundary conditions
struct HyperCubic{N}
  dims::NTuple{N,Int}
end
Base.size(l::HyperCubic) = l.dims

const Chain = HyperCubic{1}
const Square = HyperCubic{2}
const Cubic = HyperCubic{3}

struct Edge{N}
  edge::Tuple{NTuple{N,Int},NTuple{N,Int}}
  boundary::Bool
end

Base.getindex(edge::Edge, n::Int) = edge.edge[n]

Base.reverse(edge::Edge) = Edge(reverse(edge.edge), edge.boundary)

sites(lattice::HyperCubic) = (Tuple(s) for s in CartesianIndices(axes(lattice)))

function is_in_edge(site::Tuple, edge::Edge)
  if site == edge[1]
    return false
  elseif site == edge[2]
    return true
  else
    error("Site $site is not incident to edge $edge")
  end
end

function onehot_tuple(n::Integer, length::Val{N}) where {N}
  return ntuple(i -> i == n ? 1 : 0, Val(N))
end

# Obtain the neighbor in dimension `dim` in direction `dir`, for example:
# neighbor((2, 2), 2, 1) == (2, 3)
# neighbor((2, 2), 2, -1) == (2, 1)
# neighbor((2, 2), 1, -1) == (1, 2)
# neighbor((2, 3), 2, 1, lattice_size=(3, 3)) == (2, 1)
function neighbor(
  site::NTuple{N,Int}, dim::Int, dir::Int; lattice_size=typemax(eltype(site))
) where {N}
  return map(
    (a, b, d) -> mod1(a + dir * b, d), site, onehot_tuple(dim, Val(N)), lattice_size
  )
end

# Check if the edge connecting to the specified neighbor of the site
# crosses the boundary of the lattice
function isboundary(
  site::NTuple{N,Int}, dim::Int, dir::Int; lattice_size=typemax(eltype(site))
) where {N}
  return ((dir == 1) && (site[dim] == lattice_size[dim])) || ((dir == -1) && site[dim] == 1)
end

# The neighboring sites of the specified site
function filterneighbors(
  f, lattice::HyperCubic{N}, site::NTuple{N,Int}; periodic=false
) where {N}
  lattice_size = size(lattice)
  site_neighbors = Vector{NTuple{N,Int}}()
  for dim in 1:N, dir in (-1, 1)
    site_neighbor = neighbor(site, dim, dir; lattice_size=lattice_size)
    bc_condition = periodic || !(isboundary(site, dim, dir; lattice_size=lattice_size))
    if f(site, site_neighbor) && bc_condition
      push!(site_neighbors, site_neighbor)
    end
  end
  return site_neighbors
end

function neighbors(lattice::HyperCubic{N}, site::NTuple{N,Int}; periodic=false) where {N}
  return filterneighbors(≠, lattice, site; periodic=periodic)
end

function inneighbors(lattice::HyperCubic{N}, site::NTuple{N,Int}; periodic=false) where {N}
  return filterneighbors(>, lattice, site; periodic=periodic)
end

function outneighbors(lattice::HyperCubic{N}, site::NTuple{N,Int}; periodic=false) where {N}
  return filterneighbors(<, lattice, site; periodic=periodic)
end

# All of the edges connected to the vertex `site`
function incident_edges(lattice::HyperCubic{N}, site::NTuple{N,Int}) where {N}
  lattice_size = size(lattice)
  site_edges = Vector{Edge{N}}()
  for dim in 1:N, dir in (-1, 1)
    site_neighbor = neighbor(site, dim, dir; lattice_size=lattice_size)
    boundary = false
    if isboundary(site, dim, dir; lattice_size=lattice_size)
      boundary = true
    end
    edge = (site, site_neighbor)
    if dir == -1
      edge = reverse(edge)
    end
    push!(site_edges, Edge(edge, boundary))
  end
  return site_edges
end

function bonds(lattice::HyperCubic; periodic=false)
  return [
    (s, n) for s in sites(lattice) for n in outneighbors(lattice, s; periodic=periodic)
  ]
end

function bonds(lattice::Square, coord::Tuple{Colon,<:Integer})
  rowsize = lattice.dims[1]
  colsites = [(i, coord[2]) for i in 1:(rowsize - 1)]
  return [(s, (s[1] + 1, s[2])) for s in colsites]
end

function bonds(lattice::Square, coord::Tuple{<:Integer,Colon})
  colsize = lattice.dims[2]
  rowsites = [(coord[1], i) for i in 1:(colsize - 1)]
  return [(s, (s[1], s[2] + 1)) for s in rowsites]
end
