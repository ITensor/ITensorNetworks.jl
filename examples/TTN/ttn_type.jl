using ITensors
using ITensors.ContractionSequenceOptimization
using ITensorNetworks
using ITensorUnicodePlots

g = named_binary_tree(3)
s = siteinds("S=1/2", g)
ψ = TTNS(s; link_space=3)

for v in vertices(ψ)
  ψ[v] = randn!(ψ[v])
end

@visualize ψ

e = 1 => (1, 1)
ψ̃ = contract(ψ, e)

@visualize ψ̃

ψᴴ = prime(dag(ψ); sites=[])
Z = ψᴴ ⊗ ψ;

@visualize Z

# Contract across bra and ket
for v in vertices(ψ)
  global Z = contract(Z, (1, v...) => (2, v...))
end

@visualize Z

sequence = optimal_contraction_sequence(Z)

@show sequence

z = contract(Z; sequence)[]

@show z

# Contract according to `bfs_tree`.
# Currently there is a bug.
z2 = Z
source = (1, 1)
@visualize z2
for e in reverse(edges(bfs_tree(Z, source)))
  @show e
  global z2 = contract(z2, e)
  @visualize z2
end
@show z2[source][1]

nothing
