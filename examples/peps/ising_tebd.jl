using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using UnicodePlots
using Random

include("utils.jl")

Random.seed!(1234)

dims = (8, 8)
n = prod(dims)
g = named_grid(dims)
h = 0.5

@show dims

#
# DMRG comparison
#

g_dmrg = rename_vertices(g, cartesian_to_linear(dims))
ℋ_dmrg = ising(g_dmrg; h)
s_dmrg = siteinds("S=1/2", g_dmrg)
H_dmrg = MPO(ℋ_dmrg, s_dmrg)
ψ_dmrg = MPS(s_dmrg, j -> "↑")
@show inner(ψ_dmrg', H_dmrg, ψ_dmrg)
E_dmrg, ψ0_dmrg = dmrg(H_dmrg, ψ_dmrg; nsweeps=4, maxdim=10, cutoff=1e-5)
@show E_dmrg
Z_dmrg = reshape(expect(ψ0_dmrg, "Z"), dims)

display(Z_dmrg)
display(heatmap(Z_dmrg))

#
# PEPS TEBD optimization
#

s = siteinds("S=1/2", g)

ℋ = ising(g; h)

ψ_peps = ITensorNetwork(s, v -> "↑")

χ = 4
E = expect(ℋ, ψ_peps; cutoff=1e-6, maxdim=χ)
@show E

β = 10.0
Δβ = 0.1
ψ_peps = tebd(ℋ, ψ_peps; β, Δβ, cutoff=1e-6, maxdim=χ)

E_peps = expect(ℋ, ψ_peps; cutoff=1e-6, maxdim=χ)
@show E_peps

Z_peps_dict = expect("Z", ψ_peps; cutoff=1e-6, maxdim=χ)
Z_peps = [Z_peps_dict[Tuple(I)] for I in CartesianIndices(dims)]

display(Z_peps)
display(heatmap(Z_peps))
