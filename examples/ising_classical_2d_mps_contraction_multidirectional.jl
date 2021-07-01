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

space = [QN(0) => 2]
#space = 2
tn_inds = inds_network(N...; linkdims=space, addtags="S=1/2")
A = Models.local_boltzmann_weight("ising", Val(ndims); β)

tn = itensor.((A,), tn_inds)

# Project periodic boundary indices
# onto the 1 state
state = 1
tn = project_boundary(tn, state)

## function insert_projectors(tn; center, projector_center=nothing, maxdim, cutoff)
##   # For now, only support contracting from top-to-bottom
##   # and bottom-to-top down to a specified row of the network
##   @assert (center[2] == :)
##   center_row = center[1]
## 
##   if isnothing(project_center)
##     project_center = (:, size(tn) ÷ 2)
##   end
##   @assert (project_center[1] == :)
## 
##   # Approximately contract the tensor network.
##   # Outputs a Vector of boundary MPS.
##   boundary_mps_top = contract_approx(tn; alg="boundary_mps", dir="top_to_bottom", cutoff=cutoff, maxdim=maxdim)
##   boundary_mps_bottom = contract_approx(tn; alg="boundary_mps", dir="bottom_to_top", cutoff=cutoff, maxdim=maxdim)
##   # Insert approximate projectors into rows of the network
##   boundary_mps = vcat(boundary_mps_top[1:(center_row - 1)], dag.(boundary_mps_bottom[center_row:end]))
##   return insert_projectors(tn, boundary_mps; dir="top_to_bottom", projector_center=project_center)
## end

_cutoff = 1e-15
_maxdim = 100
center = (2, :)

# Outputs a tuple of the original tensor network
# and the tensors making up the projectors
tn_projected = insert_projectors(tn; center=center, maxdim=_maxdim, cutoff=_cutoff)
tn_split, Pl, Pr = tn_projected

Pl_flat = reduce(vcat, Pl)
Pr_flat = reduce(vcat, Pr)
tn_projected_flat = mapreduce(vec, vcat, (tn_split, Pl_flat, Pr_flat))

@show noncommoninds(tn_projected_flat...)
@visualize *(tn_projected_flat...) contract=false
#@visualize *(tn_split...) contract=false

@disable_warn_order begin
  @show contract(tn_projected_flat)[] / contract(vec(tn))[]
end

