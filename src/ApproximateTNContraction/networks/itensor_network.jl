
#
# Some general tools for working with networks of ITensors.
#

"""
    itensor_network(dims::Int...; linkdims)
    itensor_network(s::Array{<:Index}; linkdims)
    itensor_network(s::Array{<:Vector{<:Index}}; linkdims)

Create a tensor network on a hypercubic lattice of
dimension `dims` with link dimension `linkdims`.

Alternatively, specify the site indices with an Array `s`,
in which case the lattice will be of dimension `size(s)`.

The network will have periodic boundary conditions.
To remove the periodic boundary condiitions, use
the function `project_boundary`.
"""
function itensor_network(dims::Int...; linkdims)
  return ITensor.(inds_network(dims...; linkdims=linkdims))
end

function itensor_network(s::Array; linkdims)
  return ITensor.(inds_network(s; linkdims=linkdims))
end

"""
    boundary_projectors(tn::Matrix{ITensor}, state=1)

For a 2D tensor network, return the right and bottom boundary projectors onto
the local state `state`.
"""
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

function boundary_projectors(tn::Array{ITensor,3}, state=1)
  top = tn[1, :, :]
  bottom = tn[end, :, :]
  left = tn[:, 1, :]
  right = tn[:, end, :]
  front = tn[:, :, 1]
  back = tn[:, :, end]
  top_inds = commonind.(top, bottom)
  left_inds = commonind.(left, right)
  front_inds = commonind.(front, back)

  psi_top = ITensors.state.(top_inds, state)
  psi_left = ITensors.state.(left_inds, state)
  psi_front = ITensors.state.(front_inds, state)
  return psi_top, psi_left, psi_front
end

"""
    project_boundary(tn::Matrix{ITensor}, state=1)

Project the boundary of a periodic 2D tensor network onto 
the specified state.
"""
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

function project_boundary(tn::Array{ITensor,3}, state=1)
  Nx, Ny, Nz = size(tn)
  psi_top, psi_left, psi_front = boundary_projectors(tn, state)
  for j in 1:Ny
    for k in 1:Nz
      tn[1, j, k] = tn[1, j, k] * psi_top[j, k]
      tn[end, j, k] = tn[end, j, k] * psi_top[j, k]
    end
  end
  for i in 1:Nx
    for k in 1:Nz
      tn[i, 1, k] = tn[i, 1, k] * psi_left[i, k]
      tn[i, end, k] = tn[i, end, k] * psi_left[i, k]
    end
  end
  for i in 1:Nx
    for j in 1:Ny
      tn[i, j, 1] = tn[i, j, 1] * psi_front[i, j]
      tn[i, j, end] = tn[i, j, end] * psi_front[i, j]
    end
  end
  return tn
end

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

"""
    alllinkinds(tn)

Return a dictionary of all of the link indices of the network.
The link indices are determined by searching through the network
for tensors with indices in common with other tensors, and
the keys of the dictionary store a tuple of the sites with
the common indices.

Notice that this version will return a dictionary containing
repeated link indices, since the 
For example:
```julia
i, j, k, l = Index.((2, 2, 2, 2))
inds_network = [(i, dag(j)), (j, dag(k)), (k, dag(l)), (l, dag(i))]
tn_network = randomITensor.(inds_network)
links = allinkinds(tn_network)
links[(1, 2)] == (dag(j),)
links[(2, 1)] == (j,)
links[(2, 3)] == (dag(l),)
links[(1, 3)] # Error! In the future this may return an empty Tuple
```

Use `inlinkinds` and `outlinkinds` to return dictionaries without
repeats (such as only the link `(2, 1)` and not `(1, 2)` or vice versa).
"""
alllinkinds(tn) = filter_alllinkinds(≠, tn)
inlinkinds(tn) = filter_alllinkinds(>, tn)
outlinkinds(tn) = filter_alllinkinds(<, tn)

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

"""
    neighbors(tn, n)

From a tensor network `tn` and a site/node `n`, determine the neighbors
of the specified tensor `tn[n]` by searching for which other
tensors in the network have indices in common with `tn[n]`.

Use `inneighbors` and `outneighbors` for directed versions.
"""
function neighbors(tn, n)
  return filterneighbors(≠, tn, n)
end
inneighbors(tn, n) = filterneighbors(>, tn, n)
outneighbors(tn, n) = filterneighbors(<, tn, n)

function mapinds(f, ::typeof(linkinds), tn)
  tn′ = copy(tn)
  inds_dict = Dict()
  for n in keys(tn)
    for nn in neighbors(tn, n)
      commonindsₙ = commoninds(tn[n], tn[nn])
      newinds = []
      for i in commonindsₙ
        if !haskey(inds_dict, i)
          inds_dict[i] = f(i)
        end
        newinds = vcat(newinds, [inds_dict[i]])
      end
      tn′[n] = replaceinds(tn′[n], commonindsₙ => newinds)
    end
  end
  return tn′
end

function ITensors.prime(::typeof(linkinds), tn, args...)
  return mapinds(x -> prime(x, args...), linkinds, tn)
end

function ITensors.sim(::typeof(linkinds), tn, args...)
  return mapinds(x -> sim(x, args...), linkinds, tn)
end

function ITensors.addtags(::typeof(linkinds), tn, args...)
  return mapinds(x -> addtags(x, args...), linkinds, tn)
end

function ITensors.removetags(::typeof(linkinds), tn, args...)
  return mapinds(x -> removetags(x, args...), linkinds, tn)
end

function ITensors.prime(indices::Array{<:Index,1}, network::Array{ITensor}, n::Integer=1)
  function primeinds(tensor)
    prime_inds = [ind for ind in inds(tensor) if ind in indices]
    if (length(prime_inds) == 0)
      return tensor
    end
    return replaceinds(tensor, prime_inds => prime(prime_inds, n))
  end
  return map(x -> primeinds(x), network)
end

function ITensors.replaceinds(
  network::Union{Array{ITensor},Array{OrthogonalITensor}}, sim_dict::Dict
)
  if length(network) == 0
    return network
  end
  indices = collect(keys(sim_dict))
  function siminds(tensor)
    sim_inds = [ind for ind in inds(tensor) if ind in indices]
    if (length(sim_inds) == 0)
      return tensor
    end
    outinds = map(i -> sim_dict[i], sim_inds)
    return replaceinds(tensor, sim_inds => outinds)
  end
  return map(x -> siminds(x), network)
end

function ITensors.commoninds(n1::Array{ITensor}, n2::Array{ITensor})
  return mapreduce(a -> commoninds(a...), vcat, zip(n1, n2))
end

# Compute the sets of combiners that combine the link indices
# of the tensor network so that neighboring tensors only
# share a single larger index.
# Return a dictionary from a site to a combiner.
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

# Insert the gauge tensors `gauge` into the links of the tensor
# network `tn`.
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

# Insert the gauge tensors `gauge` into the links of the sets
# of tensor networks `tn` stored in a NamedTuple.
# TODO: is this used anywhere?
function insert_gauge(tn::NamedTuple, gauge)
  return map(x -> insert_gauge.(x, (gauge,)), tn)
end

# Split the links of an ITensor network by changing the prime levels
# or tags of pairs of links.
function split_links(H::Union{MPS,MPO}; split_tags=("" => ""), split_plevs=(0 => 1))
  left_tags, right_tags = split_tags
  left_plev, right_plev = split_plevs
  l = outlinkinds(H)
  Hsplit = copy(H)
  for bond in keys(l)
    n1, n2 = bond
    lₙ = l[bond]
    left_l_n = prime(addtags(lₙ, left_tags), left_plev)
    right_l_n = prime(addtags(lₙ, right_tags), right_plev)
    Hsplit[n1] = replaceinds(Hsplit[n1], lₙ => left_l_n)
    Hsplit[n2] = replaceinds(Hsplit[n2], lₙ => right_l_n)
  end
  return Hsplit
end

function split_links(H::Vector{ITensor}, args...; kwargs...)
  return data(split_links(MPS(H), args...; kwargs...))
end
