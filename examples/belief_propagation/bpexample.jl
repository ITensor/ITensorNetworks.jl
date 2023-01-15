using Compat
using ITensors
using Metis
using ITensorNetworks

using ITensorNetworks: construct_initial_mts, update_all_mts, get_single_site_expec

n = 4
system_dims = (n, n)
g = named_grid(system_dims)
# g = named_comb_tree(system_dims)
s = siteinds("S=1/2", g)
chi = 2

ψ = randomITensorNetwork(s; link_space=chi)

ψψ = norm_sqr_network(ψ; flatten=true, map_bra_linkinds=prime)
combiners = linkinds_combiners(ψψ)
ψψ = combine_linkinds(ψψ, combiners)

# Apply Sz to site v
v = one.(system_dims)
Oψ = copy(ψ)
Oψ[v] = apply(op("Sz", s[v]), ψ[v])
ψOψ = inner_network(ψ, Oψ; flatten=true, map_bra_linkinds=prime)
ψOψ = combine_linkinds(ψOψ, combiners)

# Get the value of sz on v via exact contraction
contract_seq = contraction_sequence(ψψ)
actual_sz = contract(ψOψ; sequence=contract_seq)[] / contract(ψψ; sequence=contract_seq)[]

println("Actual value of Sz on site " * string(v) * " is " * string(actual_sz))

niters = 20

nsites = 1
println("\nFirst " * string(nsites) * " sites form a subgraph")
mts = construct_initial_mts(ψψ, nsites; init=(I...) -> @compat allequal(I) ? 1 : 0)
@show get_single_site_expec(ψψ, mts, ψOψ, v)
mts = update_all_mts(ψψ, mts, niters)
@show get_single_site_expec(ψψ, mts, ψOψ, v)

nsites = 4
println("\nNow " * string(nsites) * " sites form a subgraph")
mts = construct_initial_mts(ψψ, nsites; init=(I...) -> @compat allequal(I) ? 1 : 0)
@show get_single_site_expec(ψψ, mts, ψOψ, v)
mts = update_all_mts(ψψ, mts, niters)
@show get_single_site_expec(ψψ, mts, ψOψ, v)
