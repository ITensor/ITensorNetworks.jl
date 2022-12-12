using ITensors
using ITensorNetworks
using ITensorUnicodePlots
using UnicodePlots
using Random

Random.seed!(1234)

ITensors.disable_warn_order()

dims = (6, 6)
n = prod(dims)
g = named_grid(dims)

h = 2.0

@show h
@show dims

s = siteinds("S=1/2", g)

#
# DMRG comparison
#

g_dmrg = rename_vertices(g, cartesian_to_linear(dims))
ℋ_dmrg = ising(g_dmrg; h)
s_dmrg = [only(s[v]) for v in vertices(s)]
H_dmrg = MPO(ℋ_dmrg, s_dmrg)
ψ_dmrg_init = MPS(s_dmrg, j -> "↑")
@show inner(ψ_dmrg_init', H_dmrg, ψ_dmrg_init)
E_dmrg, ψ_dmrg = dmrg(
  H_dmrg, ψ_dmrg_init; nsweeps=20, maxdim=[fill(10, 10); 20], cutoff=1e-8
)
@show E_dmrg
Z_dmrg = reshape(expect(ψ_dmrg, "Z"), dims)

display(Z_dmrg)
display(heatmap(Z_dmrg))

#
# PEPS TEBD optimization
#

ℋ = ising(g; h)

χ = 2

# Enable orthogonalizing the PEPS using a local gauge transformation
ortho = true

ψ_init = ITensorNetwork(s, v -> "↑")

β = 1.0
Δβ = 0.1

println("maxdim = $χ")
@show β, Δβ
@show ortho

# Contraction sequence for exactly computing expectation values
inner_sequence = reduce((x, y) -> [x, y], vec(Tuple.(CartesianIndices(dims))))

println("\nFirst run TEBD without orthogonalization")
ψ = @time tebd(
  group_terms(ℋ, g), ψ_init; β, Δβ, cutoff=1e-8, maxdim=χ, ortho=false, print_frequency=1
)

println("\nMeasure energy expectation value")
E = @time expect(ℋ, ψ; sequence=inner_sequence)
@show E

println("\nThen run TEBD with orthogonalization (more accurate)")
ψ = @time tebd(
  group_terms(ℋ, g), ψ_init; β, Δβ, cutoff=1e-8, maxdim=χ, ortho, print_frequency=1
)

println("\nMeasure energy expectation value")
E = @time expect(ℋ, ψ; sequence=inner_sequence)
@show E

println("\nMeasure magnetization")
Z_dict = @time expect("Z", ψ; sequence=inner_sequence)
Z = [Z_dict[Tuple(I)] for I in CartesianIndices(dims)]
display(Z)
display(heatmap(Z))
