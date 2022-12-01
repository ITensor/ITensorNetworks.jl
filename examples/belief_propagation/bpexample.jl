using ITensors
using Metis
using ITensorNetworks

using ITensorNetworks:
  subgraphs,
  construct_initial_mts,
  update_all_mts,
  iterate_single_site_expec,
  get_single_site_expec

n = 2
g = named_grid(n)
# g = named_comb_tree((n, n))
s = siteinds("S=1/2", g)
chi = 2

ψ = randomITensorNetwork(s; link_space=chi)

ψψ = norm_sqr_network(ψ; flatten=true, map_bra_linkinds=prime)
combiners = linkinds_combiners(ψψ)
ψψ = combine_linkinds(ψψ, combiners)

# Apply Sz to site v
# v = (1, 1)
v = 1
# Oψ = apply(op("Sz", s[v]), ψ; ortho=false)

@show ψ[2]

Oψ = copy(ψ)

@show Oψ[2]

Oψ[v] = apply(op("Sz", s[v]), ψ[v])

@show Oψ[2]

ψOψ = inner_network(ψ, Oψ; flatten=true, map_bra_linkinds=prime)
ψOψ = combine_linkinds(ψOψ, combiners)

@show Oψ[2]
@show ψψ[2]
@show ψOψ[2]

#Get the value of sz on v via exact contraction
actual_sz = contract(ψOψ)[] / contract(ψψ)[]

println("Actual value of Sz on site " * string(v) * " is " * string(actual_sz))

niters = 20

nsites = 1
println("First " * string(nsites) * " sites form a subgraph")
mts = construct_initial_mts(ψψ, nsites; init=(I...) -> allequal(I) ? 1 : 0)

@show ψψ[2]
@show ψOψ[2]

# mts = update_all_mts(ψψ, init_mts, niters)

mts = iterate_single_site_expec(deepcopy(ψψ), deepcopy(mts), niters, deepcopy(ψOψ), v)

@show ψψ[2]
@show ψOψ[2]

# nsites = 4
# println("Now " * string(nsites) * " sites form a subgraph")
# mts = construct_initial_mts(ψψ, nsites; init=(I...) -> allequal(I) ? 1 : 0)
# iterate_single_site_expec(ψψ, mts, niters, ψOψ, v)
