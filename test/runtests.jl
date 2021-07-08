using ITensors
using ITensorNetworks
using Test

using ITensorNetworks: Models, inds_network, project_boundary, contract_approx, insert_projectors, sqnorm, sqnorm_approx

@testset "ITensorNetworks.jl" begin
  model = Models.Model"ising"()
  βc = Models.critical_point(model)
  β = 1.001 * βc

  N = (3, 4)
  ndims = length(N)

  space = 2
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

  @test isempty(noncommoninds(tn_projected_flat...))

  @disable_warn_order begin
    @test contract(tn_projected_flat)[] ≈ contract(vec(tn))[]
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

  @test isempty(noncommoninds(tn_projected_flat...))

  @disable_warn_order begin
    @test contract(tn_projected_flat)[] ≈ contract(vec(tn))[]
  end
end

function peps_tensor(; linkdim, sitedim)
  # left, right, top, bottom, site
  return randn(linkdim, linkdim, linkdim, linkdim, sitedim)
end

@testset "PEPS norm approx" begin
  N = (3, 3)
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
  @test isempty(noncommoninds(sqnormψ_approx...))

  @disable_warn_order begin
    @test contract(sqnormψ_approx)[] / contract(sqnormψ)[] ≈ 1.0
  end
end

