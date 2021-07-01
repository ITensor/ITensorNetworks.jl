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
site_space = [QN(0) => site_dim]
link_space = [QN(0) => link_dim]
site_inds = siteinds("S=1/2", N...; space=site_space)
link_inds = inds_network(N...; linkdims=link_space)
@show link_inds
#tn_inds = vcat.(link_inds, site_inds)

function peps_tensor(; linkdim, sitedim)
  # left, right, top, bottom, site
  return randn(linkdim, linkdim, linkdim, linkdim, sitedim)
end

A = symmetric_peps_tensor(; linkdim=link_dim, sitedim=site_dim)
ψ = itensor.((A,), tn_inds)

# Project periodic boundary indices
# onto the 1 state
state = 1
ψ = project_boundary(ψ, state)

# Square the tensor network
ψᴴ = dag.(ψ)
ψ′ = prime(linkinds, ψ)
tn = ψ′ .* ψᴴ
# Output the link combiners as `combiner_guage`
tn = combine_linkinds(tn)

_cutoff = 1e-15
_maxdim = 100

# Contract in every directions
boundary_mps = contract_approx(tn; maxdim=_maxdim, cutoff=_cutoff)

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
@visualize *(tn_projected_flat...) contract=false pause=true

@disable_warn_order begin
  @show contract(tn_projected_flat)[] / contract(vec(tn))[]
end

#
# Insert projectors vertically (to measure e.g. properties
# in a column of the network)
#

column = 2
center = (:, column)
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
tn_projected_uncombined = insert_gauge(tn_projected_flat, combiner_gauge)
projectors_uncombined = tn_split[(length(tn) + 1):end]


