using ITensors
using ITensorNetworks
using ITensorsVisualization
using ITensorNetworks: Models, inds_network, project_boundary, sqnorm, sqnorm_approx

function peps_tensor(; linkdim, sitedim)
  # left, right, top, bottom, site
  return randn(linkdim, linkdim, linkdim, linkdim, sitedim)
end

N = (3, 4)
ndims = length(N)

site_inds = siteinds("S=1/2", N...)
link_space = 2
inds_net = inds_network(site_inds; linkdims=link_space)

A = peps_tensor(; linkdim=link_space, sitedim=dim(first(site_inds)))
ψ = itensor.((A,), inds_net)

# Project periodic boundary indices
# onto the 1 state
state = 1
ψ = project_boundary(ψ, state)

row = 2
center = (row, :)
cutoff_ = 1e-15
maxdim_ = 100

sqnormψ = sqnorm(ψ)
sqnormψ_approx = sqnorm_approx(ψ; center=center, cutoff=cutoff_, maxdim=maxdim_)
@show noncommoninds(sqnormψ_approx...)
@visualize *(sqnormψ_approx...) contract=false

@disable_warn_order begin
  @show contract(sqnormψ_approx)[] / contract(sqnormψ)[]
end

