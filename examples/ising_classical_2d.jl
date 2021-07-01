using ITensors
using ITensorNetworks
using ITensorsVisualization

using ITensorNetworks: inds_network, project_boundary, contract_approx, insert_projectors
using ITensorNetworks: Models

model = Models.Model"ising"()
βc = Models.critical_point(model)
β = 1.001 * βc
@show β / βc

N = (3, 4)
ndims = length(N)

space = [QN(0) => 2]
#space = 2
tn_inds = inds_network(N...; linkdims=space, addtags="S=1/2")
A = Models.local_boltzmann_weight("ising", Val(ndims); β)

tn = itensor.((A,), tn_inds)

# Project periodic boundary indices
# onto the 1 state
state = 1
tn = project_boundary(tn, state)

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

