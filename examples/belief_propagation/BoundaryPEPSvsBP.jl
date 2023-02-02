using ITensors
using ITensorNetworks
using Random
using Statistics
using NPZ
using ITensorNetworks:
  contract_boundary_mps,
  construct_initial_mts,
  update_all_mts,
  get_single_site_expec,
  contract_inner
using Compat

function get_onsite_Sz_approx_BoundaryMPS(
  g, s::IndsNetwork, ψ::ITensorNetwork, lx::Int64, ly::Int64, v::Tuple; kwargs...
)
  ψψ = norm_sqr_network(ψ; flatten=true, map_bra_linkinds=prime)
  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)

  Zapprox = contract_boundary_mps(ψψ; kwargs...)

  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])

  ψOψ = inner_network(ψ, Oψ; flatten=true, map_bra_linkinds=prime)
  ψOψ = combine_linkinds(ψOψ, combiners)
  O = contract_boundary_mps(ψOψ; kwargs...)

  return O[1] / Zapprox[1]
end

function get_onsite_Sz_approx_GBP(
  g,
  ψ::ITensorNetwork,
  s::IndsNetwork,
  chi::Int64,
  lx::Int64,
  ly::Int64,
  nx::Int64,
  ny::Int64,
  niters::Int64,
  v::Tuple,
)
  ψψ = norm_sqr_network(ψ; flatten=true, map_bra_linkinds=prime)
  combiners = linkinds_combiners(ψψ)
  ψψ = combine_linkinds(ψψ, combiners)

  Oψ = copy(ψ)
  Oψ[v] = apply(op("Sz", s[v]), ψ[v])
  ψOψ = inner_network(ψ, Oψ; flatten=true, map_bra_linkinds=prime)
  ψOψ = combine_linkinds(ψOψ, combiners)

  mts = construct_initial_mts(ψψ, nx * ny; init=(I...) -> allequal(I) ? 1 : 0)

  mts = update_all_mts(ψψ, mts, niters)

  return get_single_site_expec(ψψ, mts, ψOψ, v)
end

function benchmark(
  g,
  psi::ITensorNetwork,
  s::IndsNetwork,
  chi::Int64,
  chis,
  niters::Int64,
  nxs,
  nys,
  lx::Int64,
  ly::Int64,
  v::Tuple,
)
  n_vertices = length(vertices(psi))

  sz_Boundary = zeros((length(chis)))
  sz_Boundary_Times = zeros((length(chis)))
  for i in 1:length(chis)
    println("Chi is " * string(chis[i]) * ", computing Boundary MPS Solution")
    sz_Boundary_Times[i] = @elapsed sz_Boundary[i] = get_onsite_Sz_approx_BoundaryMPS(
      g, s, psi, lx, ly, v; maxdim=chis[i]
    )
    println("Time Taken is " * string(sz_Boundary_Times[i]) * " Seconds.")
  end

  sz_GBP = zeros((length(nxs)))
  sz_GBP_Times = zeros((length(nxs)))
  for i in 1:length(nxs)
    println(
      "Nx is " *
      string(nxs[i]) *
      " ,Ny is " *
      string(nys[i]) *
      " computing Boundary MPS Solution",
    )
    sz_GBP_Times[i] = @elapsed sz_GBP[i] = get_onsite_Sz_approx_GBP(
      g, psi, s, chi, lx, ly, nxs[i], nys[i], niters, v
    )
    println("Time Taken is " * string(sz_GBP_Times[i]) * " Seconds.")
  end

  return sz_Boundary, sz_Boundary_Times, sz_GBP, sz_GBP_Times
end

Random.seed!(1454)

#LETS TEST SOME BOUNDARY MPS STUFF AGAINST BP

lx, ly = 4, 4
g = named_grid((lx, ly))
chi = 2
chi_max = 2
s = siteinds("S=1/2", g)
psi = randomITensorNetwork(s; link_space=chi)
niters = 20
nxs, nys = [1, 1, 2, 8], [1, 2, 2, 1]
chis = [1, 2, 4, 8, 16, 32, 64, 128]
v = (1, 1)

sz_Boundary, sz_Boundary_Times, sz_GBP, sz_GBP_Times = benchmark(
  g, psi, s, chi, chis, niters, nxs, nys, lx, ly, v
)

display(sz_Boundary)
display(sz_GBP)

#npzwrite("Data/RandTensorCalc2DGridChi"*string(chi)*"Lx"*string(lx)*"Ly"*string(ly)*"Vx"*string(v[1])*"Vy"*string(v[2])*".npz", nxs = nxs, nys = nys, chis = chis, sz_Boundary = sz_Boundary, sz_GBP = sz_GBP, sz_Boundary_Times = sz_Boundary_Times, sz_GBP_Times = sz_GBP_Times)
