
#
# ITensors.jl extensions
#

# Generalize siteind to n-dimensional lattice
function ITensors.siteind(st::SiteType, N1::Integer, N2::Integer, Ns::Integer...; kwargs...)
  s = siteind(st; kwargs...)
  if !isnothing(s)
    ts = "n1=$N1,n2=$N2"
    for i in eachindex(Ns)
      ts *= ",n$(i + 2)=$(Ns[i])"
    end
    return addtags(s, ts)
  end
  isnothing(s) && error(space_error_message(st))
end

# Generalize siteinds to n-dimensional lattice
function ITensors.siteinds(str::AbstractString, N1::Integer, N2::Integer, Ns::Integer...; kwargs...)
  st = SiteType(str)
  return [siteind(st, ns...) for ns in Base.product(1:N1, 1:N2, UnitRange.(1, Ns)...)]
end

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

function coordinate_tag(n)
  str = replace("$n", ")" => "")
  str = replace(str, "(" => "")
  str = replace(str, " " => "")
  if length(n) > 1
    str = replace(str, "," => ".")
  else
    str = replace(str, "," => "")
  end
  return str
end

function link_tag(n1, n2)
  link_string = "$(coordinate_tag(n1))↔$(coordinate_tag(n2))"
  start_ind = nextind(link_string, 0, 1)
  stop_ind = min(ncodeunits(link_string), nextind(link_string, 0, 8))
  link_string = link_string[start_ind:stop_ind]
  return TagSet(link_string)
end

function ITensors.linkinds(lattice::HyperCubic; linkdims, addtags=ts"")
  dims = size(lattice)
  N = length(dims)
  linkinds_dict = Dict{Edge{N},Index{typeof(linkdims)}}()
  for n in sites(lattice), edge_n in incident_edges(lattice, n)
    l = Index(linkdims; tags=ITensors.addtags(link_tag(edge_n.edge...), addtags))
    get!(linkinds_dict, edge_n, l)
  end
  return linkinds_dict
end

using ITensors: data

function Base.isless(i1::Index, i2::Index)
  return isless((id(i1), plev(i1), tags(i1), dir(i1)), (id(i2), plev(i2), tags(i2), dir(i2)))
end

Base.isless(ts1::TagSet, ts2::TagSet) = isless(data(ts1), data(ts2))

function is_in_edge(site::Tuple, edge::Edge)
  if site == edge[1]
    return false
  elseif site == edge[2]
    return true
  else
    error("Site $site is not incident to edge $edge")
  end
end

function get_link_ind(linkinds_dict::Dict, edge::Edge, site::Tuple)
  l = linkinds_dict[edge]
  return is_in_edge(site, edge) ? dag(l) : l
end

# A network of link indices for a HyperCubic lattice, with
# no site indices.
function inds_network(dims::Int...; linkdims, kwargs...)
  site_inds = fill(Index{typeof(linkdims)}[], dims)
  return inds_network(site_inds; linkdims=linkdims, kwargs...)
end

function inds_network(site_inds::Array{<:Index,N}; kwargs...) where {N}
  return inds_network(map(x -> [x], site_inds); kwargs...)
end

# A network of link indices for a HyperCubic lattice, with
# site indices specified.
function inds_network(site_inds::Array{<:Vector{<:Index},N}; linkdims, addtags=ts"") where {N}
  dims = size(site_inds)
  lattice = HyperCubic(dims)
  linkinds_dict = linkinds(lattice; linkdims, addtags)
  inds = Array{Vector{Index{typeof(linkdims)}},N}(undef, dims)
  for n in sites(lattice)
    inds_n = [get_link_ind(linkinds_dict, edge_n, n) for edge_n in incident_edges(lattice, n)]
    inds[n...] = append!(inds_n, site_inds[n...])
  end
  return inds
end

function itensor_network(dims::Int...; linkdims)
  return ITensor.(inds_network(dims...; linkdims))
end

function onehot_tuple(n::Integer, length::Val{N}) where {N}
  return ntuple(i -> i == n ? 1 : 0, Val(N))
end

# Obtain the neighbor in dimension `dim` in direction `dir`, for example:
# neighbor((2, 2), 2, 1) == (2, 3)
# neighbor((2, 2), 2, -1) == (2, 1)
# neighbor((2, 2), 1, -1) == (1, 2)
# neighbor((2, 3), 2, 1, lattice_size=(3, 3)) == (2, 1)
function neighbor(site::NTuple{N,Int}, dim::Int, dir::Int; lattice_size=typemax(eltype(site))) where {N}
  return map((a, b, d) -> mod1(a + dir * b, d), site, onehot_tuple(dim, Val(N)), lattice_size)
end

#function isboundary(site::dir::Int, 
# Check if the edge connecting to the specified neighbor of the site 
# crosses the boundary of the lattice
function isboundary(site::NTuple{N,Int}, dim::Int, dir::Int; lattice_size=typemax(eltype(site))) where {N}
  return ((dir == 1) && (site[dim] == lattice_size[dim])) || ((dir == -1) && site[dim] == 1)
end

# The neighboring sites of the specified site
function neighbors(lattice::HyperCubic{N}, site::NTuple{N,Int}) where {N}
  lattice_size = size(lattice)
  site_neighbors = Vector{NTuple{N,Int}}()
  for dim in 1:N, dir in (-1, 1)
    site_neighbor = neighbor(site, dim, dir; lattice_size=lattice_size)
    push!(site_neighbors, neighbor)
  end
  return site_neighbors
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

bonds(lattice::HyperCubic) = ((s, n) for s in sites(lattice) for n in directed_neighbors(lattice, s))

function unique_bonds(lattice::HyperCubic)
  # Turn into Sets so order doesn't matter
  bond_sets = [Set(b) for b in bonds(l)]
  unique!(bond_sets)
  bond_tuples = map(Tuple, bond_sets)
  sort!(bond_tuples)
  return bond_tuples
end

#
# Contraction methods
#

# Return the right and bottom boundary projectors onto
# the local state `state`
function boundary_projectors(tn::Matrix{ITensor}, state=1)
  top_row = tn[1, :]
  bottom_row = tn[end, :]
  left_column = tn[:, 1]
  right_column = tn[:, end]
  bottom_boundary_inds = commonind.(bottom_row, top_row)
  right_boundary_inds = commonind.(right_column, left_column)
  ψr = ITensors.state.(right_boundary_inds, state)
  ψb = ITensors.state.(bottom_boundary_inds, state)
  return ψr, ψb
end

function project_boundary(tn::Matrix{ITensor}, state=1)
  Nx, Ny = size(tn)
  ψr, ψb = boundary_projectors(tn, state)
  for n in 1:Nx
    tn[n, 1] = tn[n, 1] * ψr[n]
    tn[n, end] = tn[n, end] * dag(ψr[n])
  end
  for n in 1:Ny
    tn[1, n] = tn[1, n] * ψb[n]
    tn[end, n] = tn[end, n] * dag(ψb[n])
  end
  return tn
end

Base.keytype(m::MPS) = keytype(data(m))

indtype(i::Index) = typeof(i)
indtype(T::Type{<:Index}) = T
indtype(is::Tuple{Vararg{<:Index}}) = eltype(is)
indtype(is::Vector{<:Index}) = eltype(is)
indtype(A::ITensor...) = indtype(inds.(A))

indtype(tn1, tn2) = promote_type(indtype(tn1), indtype(tn2))
indtype(tn) = mapreduce(indtype, promote_type, tn)

function filter_alllinkinds(f, tn)
  linkinds = Dict{Tuple{keytype(tn),keytype(tn)},Vector{indtype(tn)}}()
  for n in keys(tn), m in keys(tn)
   if f(n, m)
     is = commoninds(tn[n], tn[m])
     if !isempty(is)
       linkinds[(n, m)] = is
     end
   end
 end
 return linkinds
end
alllinkinds(tn) = filter_alllinkinds(≠, tn)
inlinkinds(tn) = filter_alllinkinds(>, tn)
outlinkinds(tn) = filter_alllinkinds(<, tn)

function split_links(H::Union{MPS,MPO}; split_tags=("" => ""), split_plevs=(0 => 1))
  left_tags, right_tags = split_tags
  left_plev, right_plev = split_plevs
  l = outlinkinds(H)
  Hsplit = copy(H)
  for bond in keys(l)
    n1, n2 = bond
    lₙ = l[bond]
    left_l_n = setprime(addtags(lₙ, left_tags), left_plev)
    right_l_n = setprime(addtags(lₙ, right_tags), right_plev)
    Hsplit[n1] = replaceinds(Hsplit[n1], lₙ => left_l_n)
    Hsplit[n2] = replaceinds(Hsplit[n2], lₙ => right_l_n)
  end
  return Hsplit
end

split_links(H::Vector{ITensor}, args...; kwargs...) = data(split_links(MPS(H), args...; kwargs...))

function insert_projectors(H, psi, projs)
  N = length(H)
  Hpsi = Vector{ITensor}(undef, N)
  Hpsi[1] = psi[1] * H[1] * projs[1][1]
  for n in 2:(N - 1)
    Hpsi[n] = projs[n - 1][2] * psi[n] * H[n] * projs[n][1]
  end
  Hpsi[N] = projs[N - 1][2] * psi[N] * H[N]
  return Hpsi
end

# Compute the truncation projectors for the network,
# contracting from top to bottom
function truncation_projectors(tn::Matrix{ITensor}; maxdim=maxdim_arg(tn), cutoff=1e-8, split_tags=("" => ""), split_plevs=(0 => 1))
  nrows, ncols = size(tn)
  U = Matrix{ITensor}(undef, nrows - 3, ncols - 1)
  Ud = copy(U)
  tn_split = copy(tn)
  if nrows ≤ 3
    return tn_split, U, Ud
  end
  psi = tn[1, :]
  psi_split = split_links(psi; split_tags=split_tags, split_plevs=split_plevs)
  tn_split[1, :] .= psi_split
  for n in 1:(nrows - 3)
    H = tn[n + 1, :]
    projs = truncation_projectors(H, psi; split_tags=split_tags, split_plevs=split_plevs, cutoff=cutoff)
    U[n, :] = first.(projs)
    Ud[n, :] = last.(projs)
    if n < size(U, 1)
      for m in 1:size(U, 2)
        U[n, m], Ud[n, m] = split_links([U[n, m], Ud[n, m]]; split_tags=split_tags, split_plevs=split_plevs)
      end
    end
    H_split = split_links(H; split_tags=split_tags, split_plevs=split_plevs)
    tn_split[n + 1, :] .= H_split
    psi = insert_projectors(H_split, psi_split, projs)
    psi_split = split_links(psi; split_tags=split_tags, split_plevs=split_plevs)
  end
  return tn_split, U, Ud
end

struct ContractionAlgorithm{algorithm} end

ContractionAlgorithm(s::AbstractString) = ContractionAlgorithm{Symbol(s)}()

macro ContractionAlgorithm_str(s)
  :(ContractionAlgorithm{$(Expr(:quote, Symbol(s)))})
end

Base.@kwdef struct BoundaryMPS
  top::Vector{MPS}=MPS[]
  bottom::Vector{MPS}=MPS[]
  left::Vector{MPS}=MPS[]
  right::Vector{MPS}=MPS[]
end

struct BoundaryMPSDir{dir} end

BoundaryMPSDir(s::AbstractString) = BoundaryMPSDir{Symbol(s)}()

macro BoundaryMPSDir_str(s)
  :(BoundaryMPSDir{$(Expr(:quote, Symbol(s)))})
end

function contract_approx(tn::Matrix{ITensor}, alg::ContractionAlgorithm"boundary_mps",
                         dirs::BoundaryMPSDir"top_to_bottom"; cutoff, maxdim)
  nrows, ncols = size(tn)
  boundary_mps = Vector{MPS}(undef, nrows - 1)
  x = MPS(tn[1, :])
  boundary_mps[1] = orthogonalize(x, ncols)
  for nrow in 2:(nrows - 1)
    A = MPO(tn[nrow, :])
    x = contract(A, x; cutoff=cutoff, maxdim=maxdim)
    boundary_mps[nrow] = orthogonalize(x, ncols)
  end
  return BoundaryMPS(top=boundary_mps)
end

function contract_approx(tn::Matrix{ITensor}, alg::ContractionAlgorithm"boundary_mps",
                         dirs::BoundaryMPSDir"bottom_to_top"; cutoff, maxdim)
  tn = rot180(tn)
  tn = reverse(tn; dims=2)
  boundary_mps_top = contract_approx(tn, alg, BoundaryMPSDir"top_to_bottom"(); cutoff=cutoff, maxdim=maxdim)
  return BoundaryMPS(bottom=reverse(boundary_mps_top.top))
end

function contract_approx(tn::Matrix{ITensor}, alg::ContractionAlgorithm"boundary_mps",
                         dirs::BoundaryMPSDir"left_to_right"; cutoff, maxdim)
  tn = rotr90(tn)
  boundary_mps = contract_approx(tn, alg, BoundaryMPSDir"top_to_bottom"(); cutoff=cutoff, maxdim=maxdim)
  return BoundaryMPS(left=boundary_mps.top)
end

function contract_approx(tn::Matrix{ITensor}, alg::ContractionAlgorithm"boundary_mps",
                         dirs::BoundaryMPSDir"right_to_left"; cutoff, maxdim)
  tn = rotr90(tn)
  boundary_mps = contract_approx(tn, alg, BoundaryMPSDir"bottom_to_top"(); cutoff=cutoff, maxdim=maxdim)
  return BoundaryMPS(right=boundary_mps.bottom)
end

function contract_approx(tn::Matrix{ITensor}, alg::ContractionAlgorithm"boundary_mps",
                         dirs::BoundaryMPSDir"all"; cutoff, maxdim)
  mps_top = contract_approx(tn, alg, BoundaryMPSDir"top_to_bottom"(); cutoff=cutoff, maxdim=maxdim)
  mps_bottom = contract_approx(tn, alg, BoundaryMPSDir"bottom_to_top"(); cutoff=cutoff, maxdim=maxdim)
  mps_left = contract_approx(tn, alg, BoundaryMPSDir"left_to_right"(); cutoff=cutoff, maxdim=maxdim)
  mps_right = contract_approx(tn, alg, BoundaryMPSDir"right_to_left"(); cutoff=cutoff, maxdim=maxdim)
  return BoundaryMPS(top=mps_top.top, bottom=mps_bottom.bottom, left=mps_left.left, right=mps_right.right)
end

function contract_approx(tn::Matrix{ITensor}, alg::ContractionAlgorithm"boundary_mps";
                         dirs, cutoff, maxdim)
  return contract_approx(tn, alg, BoundaryMPSDir(dirs); cutoff=cutoff, maxdim=maxdim)
end

# Compute the truncation projectors for the network,
# contracting from top to bottom
function contract_approx(tn::Matrix{ITensor}; alg="boundary_mps",
                         dirs="all", cutoff=1e-8, maxdim=maxdim_arg(tn))
  return contract_approx(tn, ContractionAlgorithm(alg); dirs=dirs, cutoff=cutoff, maxdim=maxdim)
end

function insert_projectors(tn::Matrix{ITensor}, boundary_mps::Vector{MPS}; dirs, projector_center)
  return insert_projectors(tn, boundary_mps, BoundaryMPSDir(dirs); projector_center=projector_center)
end

get_itensor(x::MPS, n::Int) = n in 1:length(x) ? x[n] : ITensor()

Base.reverse(x::MPS) = MPS(reverse(x.data))

# From an MPS, create a 1-site projector onto the MPS basis
function projector(x::MPS, projector_center)
  # Gauge the boundary MPS towards the projector_center column
  x = orthogonalize(x, projector_center)

  l = commoninds(get_itensor(x, projector_center - 1), x[projector_center])
  r = commoninds(get_itensor(x, projector_center + 1), x[projector_center])

  uₗ = x[1:(projector_center - 1)]
  uᵣ = reverse(x[(projector_center + 1):end])
  nₗ = length(uₗ)
  nᵣ = length(uᵣ)

  uₗᴴ = dag.(uₗ)
  uᵣᴴ = dag.(uᵣ)

  uₗ′ = reverse(prime.(uₗ))
  uᵣ′ = reverse(prime.(uᵣ))

  if !isempty(uₗ′)
    uₗ′[1] = replaceinds(uₗ′[1], l' => l)
  end
  if !isempty(uᵣ′)
    uᵣ′[1] = replaceinds(uᵣ′[1], r' => r)
  end

  Pₗ = vcat(uₗᴴ, uₗ′)
  Pᵣ = vcat(uᵣᴴ, uᵣ′)
  return Pₗ, Pᵣ
end

default_projector_center(tn::Matrix) = (:, (size(tn, 2) + 1) ÷ 2)

function split_network(tn::Matrix{ITensor}; projector_center=default_projector_center(tn))
  @assert (projector_center[1] == :)
  tn_split = copy(tn)
  nrows, ncols = size(tn)
  @assert length(projector_center) == 2
  @assert projector_center[1] == Colon()
  projector_center_cols = projector_center[2]
  for ncol in 1:ncols
    if ncol ∉ projector_center_cols
      tn_split[:, ncol] = split_links(tn_split[:, ncol])
    end
  end
  return tn_split
end

function insert_projectors(tn::Matrix{ITensor}, boundary_mps::Vector{MPS},
                           dirs::BoundaryMPSDir"top_to_bottom"; projector_center)
  nrows, ncols = size(tn)
  @assert length(boundary_mps) == nrows - 1
  @assert all(x -> length(x) == ncols, boundary_mps)
  @assert length(projector_center) == 2
  @assert (projector_center[1] == :)
  projector_center_cols = projector_center[2]
  projectors = [projector(x, projector_center_cols) for x in boundary_mps]
  projectors_left = first.(projectors)
  projectors_right = last.(projectors)
  tn_split = split_network(tn; projector_center=projector_center)
  return tn_split, projectors_left, projectors_right
end

function insert_projectors(tn, boundary_mps::BoundaryMPS; center, projector_center=nothing)
  top_bottom = true
  if (center[1] == :)
    center = reverse(center)
    tn = rotr90(tn)
    top_bottom = false
  end

  @assert (center[2] == :)
  center_row = center[1]

  if isnothing(projector_center)
    projector_center = default_projector_center(tn)
  end
  @assert (projector_center[1] == :)

  # Approximately contract the tensor network.
  # Outputs a Vector of boundary MPS.
  #boundary_mps_top = contract_approx(tn; alg="boundary_mps", dirs="top_to_bottom", cutoff=cutoff, maxdim=maxdim)
  #boundary_mps_bottom = contract_approx(tn; alg="boundary_mps", dirs="bottom_to_top", cutoff=cutoff, maxdim=maxdim)

  boundary_mps_top = top_bottom ? boundary_mps.top : boundary_mps.left
  boundary_mps_bottom = top_bottom ? boundary_mps.bottom : boundary_mps.right

  # Insert approximate projectors into rows of the network
  boundary_mps_tot = vcat(boundary_mps_top[1:(center_row - 1)], dag.(boundary_mps_bottom[center_row:end]))
  tn_split, projectors_left, projectors_right = insert_projectors(tn, boundary_mps_tot; dirs="top_to_bottom", projector_center=projector_center)

  if !top_bottom
    tn_split = rotl90(tn_split)
  end
  return tn_split, projectors_left, projectors_right
end

# TODO: Delete this. Better to call contract_approx seperately
# and cache the results so they can be reused in different
# calls to `insert_projectors` that have different values of `center`
function insert_projectors(tn; center, projector_center=nothing, maxdim, cutoff)
  # For now, only support contracting from top-to-bottom
  # and bottom-to-top down to a specified row of the network
  top_bottom = true
  if (center[1] == :)
    center = reverse(center)
    tn = rotr90(tn)
    top_bottom = false
  end

  @assert (center[2] == :)
  center_row = center[1]

  if isnothing(projector_center)
    projector_center = (:, (size(tn, 2) + 1) ÷ 2)
  end
  @assert (projector_center[1] == :)

  # Approximately contract the tensor network.
  # Outputs a Vector of boundary MPS.
  boundary_mps_top = contract_approx(tn; alg="boundary_mps", dirs="top_to_bottom", cutoff=cutoff, maxdim=maxdim)
  boundary_mps_bottom = contract_approx(tn; alg="boundary_mps", dirs="bottom_to_top", cutoff=cutoff, maxdim=maxdim)
  # Insert approximate projectors into rows of the network
  boundary_mps = vcat(boundary_mps_top[1:(center_row - 1)], dag.(boundary_mps_bottom[center_row:end]))
  tn_split, projectors_left, projectors_right = insert_projectors(tn, boundary_mps; dirs="top_to_bottom", projector_center=projector_center)

  if !top_bottom
    tn_split = rotl90(tn_split)
  end
  return tn_split, projectors_left, projectors_right
end

function filterneighbors(f, tn, n)
  neighbors_tn = keytype(tn)[]
  tnₙ = tn[n]
  for m in keys(tn)
    if f(n, m) && hascommoninds(tnₙ, tn[m])
      push!(neighbors_tn, m)
    end
  end
  return neighbors_tn
end
neighbors(tn, n) = filterneighbors(≠, tn, n)
inneighbors(tn, n) = filterneighbors(>, tn, n)
outneighbors(tn, n) = filterneighbors(<, tn, n)

function mapinds(f, ::typeof(linkinds), tn)
  tn′ = copy(tn)
  for n in keys(tn)
    for nn in neighbors(tn, n)
      commonindsₙ = commoninds(tn[n], tn[nn])
      tn′[n] = replaceinds(tn′[n], commonindsₙ => f(commonindsₙ))
    end
  end
  return tn′
end

ITensors.prime(::typeof(linkinds), tn, args...) = mapinds(x -> prime(x, args...), linkinds, tn)
ITensors.addtags(::typeof(linkinds), tn, args...) = mapinds(x -> addtags(x, args...), linkinds, tn)

# Return a dictionary from a site to a combiner
function combiners(::typeof(linkinds), tn)
  Cs = Dict(keys(tn) .=> (ITensor[] for _ in keys(tn)))
  for n in keys(tn)
    for nn in inneighbors(tn, n)
      commonindsₙ = commoninds(tn[n], tn[nn])
      C = combiner(commonindsₙ)
      push!(Cs[n], C)
      push!(Cs[nn], dag(C))
    end
  end
  return Cs
end

function contraction_cache_top(tn, boundary_mps::Vector{MPS}, n)
  tn_cache = fill(ITensor(1.0), size(tn))
  for nrow in 1:size(tn, 1)
    if nrow == n
      tn_cache[nrow, :] .= boundary_mps[nrow]
    elseif nrow > n
      tn_cache[nrow, :] = tn[nrow, :]
    end
  end
  return tn_cache
end

function contraction_cache_top(tn, boundary_mps::Vector{MPS})
  tn_cache_top = Vector{typeof(tn)}(undef, length(boundary_mps))
  for n in 1:length(boundary_mps)
    tn_cache_top[n] = contraction_cache_top(tn, boundary_mps, n)
  end
  return tn_cache_top
end

function contraction_cache_bottom(tn, boundary_mps::Vector{MPS})
  tn_top = rot180(tn)
  tn_top = reverse(tn_top; dims=2)
  boundary_mps_top = reverse(boundary_mps)
  tn_cache_top = contraction_cache_top(tn_top, boundary_mps_top)
  tn_cache = reverse.(tn_cache_top; dims=2)
  tn_cache = rot180.(tn_cache)
  return tn_cache
end

function contraction_cache_left(tn, boundary_mps::Vector{MPS})
  tn_top = rotr90(tn)
  tn_cache_top = contraction_cache_top(tn_top, boundary_mps)
  tn_cache = rotl90.(tn_cache_top)
  return tn_cache
end

function contraction_cache_right(tn, boundary_mps::Vector{MPS})
  tn_bottom = rotr90(tn)
  tn_cache_bottom = contraction_cache_bottom(tn_bottom, boundary_mps)
  tn_cache = rotl90.(tn_cache_bottom)
  return tn_cache
end

function contraction_cache(tn, boundary_mps)
  cache_top = contraction_cache_top(tn, boundary_mps.top)
  cache_bottom = contraction_cache_bottom(tn, boundary_mps.bottom)
  cache_left = contraction_cache_left(tn, boundary_mps.left)
  cache_right = contraction_cache_right(tn, boundary_mps.right)
  return (top=cache_top, bottom=cache_bottom, left=cache_left, right=cache_right)
end

function insert_gauge(tn::NamedTuple, gauge)
  return map(x -> insert_gauge.(x, (gauge,)), tn)
end

function insert_gauge(tn, gauge)
  tn′ = copy(tn)
  for n in keys(gauge)
    for g in gauge[n]
      if hascommoninds(tn′[n], g)
        tn′[n] *= g
      end
    end
  end
  return tn′
end

function boundary_mps_top(tn)
  return [MPS(tn[n][n, :]) for n in 1:length(tn)]
end

function boundary_mps_bottom(tn)
  tn_top = rot180.(tn)
  tn_top = reverse.(tn_top; dims=2)
  boundary_mps = boundary_mps_top(tn_top)
  return reverse(boundary_mps)
end

function boundary_mps_left(tn)
  tn_top = rotr90.(tn)
  return boundary_mps_top(tn_top)
end

function boundary_mps_right(tn)
  tn_bottom = rotr90.(tn)
  return boundary_mps_bottom(tn_bottom)
end

function boundary_mps(tn::NamedTuple)
  _boundary_mps_top = boundary_mps_top(tn.top)
  _boundary_mps_bottom = boundary_mps_bottom(tn.bottom)
  _boundary_mps_left = boundary_mps_left(tn.left)
  _boundary_mps_right = boundary_mps_right(tn.right)
  return BoundaryMPS(top=_boundary_mps_top, bottom=_boundary_mps_bottom, left=_boundary_mps_left, right=_boundary_mps_right)
end

# Return a network that when contracted equals the
# squared norm of the input network
function sqnorm(ψ::Matrix{ITensor})
  # Square the tensor network
  ψᴴ = addtags(linkinds, dag.(ψ), "bra")
  ψ′ = addtags(linkinds, ψ, "ket")
  return mapreduce(vec, vcat, (ψᴴ, ψ′))
end

# Return a network that approximates the squared norm of the
# input network
function sqnorm_approx(ψ::Matrix{ITensor}; center, cutoff, maxdim)
  # Square the tensor network
  ψᴴ = addtags(linkinds, dag.(ψ), "bra")
  ψ′ = addtags(linkinds, ψ, "ket")
  # TODO: implement contract(commoninds, ψ′, ψᴴ)
  tn = ψ′ .* ψᴴ

  # Contract in every direction
  combiner_gauge = combiners(linkinds, tn)
  tnᶜ = insert_gauge(tn, combiner_gauge)
  boundary_mpsᶜ = contract_approx(tnᶜ; maxdim=maxdim, cutoff=cutoff)

  tn_cacheᶜ = contraction_cache(tnᶜ, boundary_mpsᶜ)
  tn_cache = insert_gauge(tn_cacheᶜ, combiner_gauge)
  _boundary_mps = boundary_mps(tn_cache)

  #
  # Insert projectors horizontally (to measure e.g. properties
  # in a row of the network)
  #

  tn_projected = insert_projectors(tn, _boundary_mps; center=center)
  tn_split, Pl, Pr = tn_projected

  ψᴴ_split = split_network(ψᴴ)
  ψ′_split = split_network(ψ′)

  Pl_flat = reduce(vcat, Pl)
  Pr_flat = reduce(vcat, Pr)
  return mapreduce(vec, vcat, (ψᴴ_split, ψ′_split, Pl_flat, Pr_flat))
end

