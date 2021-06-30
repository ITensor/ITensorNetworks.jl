using ITensors
using ITensorNetworks
using ITensorsVisualization

using ITensorNetworks: inds_network, project_boundary, contract_approx, insert_projectors
using ITensorNetworks: Models

model = Models.Model"ising"()
βc = Models.critical_point(model)
β = 1.001 * βc
@show β / βc

N = (3, 3)
ndims = length(N)

#space = [QN(0) => 2]
space = 2
tn_inds = inds_network(N...; linkdims=space, addtags="S=1/2")
A = Models.local_boltzmann_weight("ising", Val(ndims); β)

tn = itensor.((A,), tn_inds)

# Project periodic boundary indices
# onto the 1 state
state = 1
tn = project_boundary(tn, state)

cutoff = 1e-15
_maxdim = 100
dir = "top_to_bottom"

# Approximately contract the tensor network.
# Outputs a matrix of boundary MPS.
boundary_mps_top = contract_approx(tn; alg="boundary_mps", dir=dir, cutoff=cutoff, maxdim=_maxdim)

# Insert approximate projectors into rows of the network
tn_projected = insert_projectors(tn, boundary_mps_top; dir=dir, center=(:, 2))

# Outputs a tuple of the original tensor network
# and the tensors making up the projectors
tn_split, Pl, Pr = tn_projected

Pl_flat = reduce(vcat, Pl)
Pr_flat = reduce(vcat, Pr)
tn_projected_flat = mapreduce(vec, vcat, (tn_split, Pl_flat, Pr_flat))

@show noncommoninds(tn_projected_flat...)
#@visualize *(tn_projected_flat...) contract=false
@visualize *(tn_split...) contract=false

#@disable_warn_order begin
#  @show contract(tn_projected_flat)[] / contract(vec(tn))[]
#end

