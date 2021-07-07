using ITensors
using ITensorNetworks
using ITensorsVisualization

using ITensorNetworks: inds_network, project_boundary, contract_approx, insert_projectors
using ITensorNetworks: Models

#model = Models.Model"ising"()
#βc = Models.critical_point(model)
#β = 1.001 * βc
#@show β / βc

N = (3, 3)
ndims = length(N)

site_dim = 2
link_dim = 2
site_space = site_dim #[QN(0) => site_dim]
link_space = link_dim #[QN(0) => link_dim]
site_inds = siteinds("S=1/2", N...; space=site_space)
inds_net = inds_network(site_inds; linkdims=link_space)

#tn_inds = vcat.(inds_net, site_inds)

# TODO: symmetrize
function peps_tensor(; linkdim, sitedim)
  # left, right, top, bottom, site
  return randn(linkdim, linkdim, linkdim, linkdim, sitedim)
end

A = peps_tensor(; linkdim=link_dim, sitedim=site_dim)
ψ = itensor.((A,), inds_net)

# Project periodic boundary indices
# onto the 1 state
state = 1
ψ = project_boundary(ψ, state)

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

function ITensors.prime(::typeof(linkinds), tn)
  tn′ = copy(tn)
  for n in keys(tn)
    for nn in neighbors(tn, n)
      commonindsₙ = commoninds(tn[n], tn[nn])
      tn′[n] = replaceinds(tn′[n], commonindsₙ => prime(commonindsₙ))
    end
  end
  return tn′
end

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

# Square the tensor network
ψᴴ = dag.(ψ)
ψ′ = prime(linkinds, ψ)
tn = ψ′ .* ψᴴ

_cutoff = 1e-15
_maxdim = 100

# Contract in every direction
combiner_gauge = combiners(linkinds, tn)
tnᶜ = insert_gauge(tn, combiner_gauge)
boundary_mpsᶜ = contract_approx(tnᶜ; maxdim=_maxdim, cutoff=_cutoff)

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
  cache_top = contraction_cache_top(tnᶜ, boundary_mpsᶜ.top)
  cache_bottom = contraction_cache_bottom(tnᶜ, boundary_mpsᶜ.bottom)
  cache_left = contraction_cache_left(tnᶜ, boundary_mpsᶜ.left)
  cache_right = contraction_cache_right(tnᶜ, boundary_mpsᶜ.right)
  return (top=cache_top, bottom=cache_bottom, left=cache_left, right=cache_right)
end


tn_cacheᶜ = contraction_cache(tnᶜ, boundary_mpsᶜ)
tn_cache_top_2 = insert_gauge(tn_cacheᶜ.top[2], combiner_gauge)

## function insert_gauge_top_bottom(boundary_mps::ITensorNetworks.BoundaryMPS, gauge)
##   for n in keys(gauge)
##     n1, n2 = Tuple(n)
##     @show n
##     mps_top_n1 = get(boundary_mps.top, n1, nothing)
##     mps_bottom_n1 = get(boundary_mps.bottom, n1, nothing)
##     for g in gauge[n]
##       if !isnothing(mps_top_n1)
##         @show commoninds(mps_top_n1[n2], g)
##       end
##       if !isnothing(mps_bottom_n1)
##         @show commoninds(mps_bottom_n1[n2], g)
##       end
##     end
##   end
##   return nothing
## end

#boundary_mps = insert_gauge_top_bottom(boundary_mpsᶜ, combiner_gauge)

#
# Insert projectors horizontally (to measure e.g. properties
# in a row of the network)
#

row = 2
center = (row, :)
tn_projected = insert_projectors(tn, boundary_mps; center=center)
tn_split, Pl, Pr = tn_projected

Pl_flat = reduce(vcat, Pl)
Pr_flat = reduce(vcat, Pr)
tn_projected_flat = mapreduce(vec, vcat, (tn_split, Pl_flat, Pr_flat))

@show noncommoninds(tn_projected_flat...)
@visualize *(tn_projected_flat...) contract=false

@disable_warn_order begin
  @show contract(tn_projected_flat)[] / contract(vec(tn))[]
end

# Uncombine the projector indices
#tn_projected_uncombined = insert_gauge(tn_projected_flat, combiner_gauge)
#projectors_uncombined = tn_split[(length(tn) + 1):end]


