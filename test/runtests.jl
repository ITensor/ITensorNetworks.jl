using ITensors
using ITensorNetworks
using Test

using ITensorNetworks: inds_network, project_boundary, truncation_projectors
using ITensorNetworks: Models

@testset "ITensorNetworks.jl" begin
  model = Models.Model"ising"()
  βc = Models.critical_point(model)
  β = 1.001 * βc

  N = (4, 4)
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

  cutoff = 1e-15
  _maxdim = 100
  split_tags = "U" => "Ud"

  tn_split, U, Ud = truncation_projectors(tn; cutoff=cutoff, maxdim=_maxdim, split_tags=split_tags)

  tn_projected = [tn_split, U, Ud]
  tn_projected_flat = vcat(vec.(tn_projected)...)

  @disable_warn_order begin
    @test contract(tn_projected_flat)[] ≈ contract(vec(tn))[] rtol=1e-5
  end

end
