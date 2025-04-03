@eval module $(gensym())
using ITensorNetworks: BilinearFormNetwork, ITensorNetwork, random_tensornetwork, siteinds, subgraph, ttn, inner, truncate, maximize_bilinearform, union_all_inds
using ITensorNetworks.ModelHamiltonians: heisenberg
using Graphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using SplitApplyCombine: group
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @test_broken, @testset
using ITensors: apply, dag, delta, prime


@testset "Maximise BilinearForm" for elt in (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  )
    begin

    rng = StableRNG(1234)

    g = named_comb_tree((3,2))
    s = siteinds("S=1/2", g)

    #One-site truncation
    a = random_tensornetwork(rng, elt, s; link_space = 3)
    b = truncate(a; maxdim_init = 3)
    f = inner(a, b; alg = "exact") / sqrt(inner(a, a; alg = "exact") * inner(b, b; alg = "exact"))
    @test f * conj(f) ≈ 1.0 atol = 10*eps(real(elt))

    #Two-site truncation
    a = random_tensornetwork(rng, elt, s; link_space = 3)
    b = truncate(a; maxdim_init = 1, solver_kwargs= (; maxdim = 3, cutoff = 1e-16, nsites = 2, tolerance = 1e-8))
    f = inner(a, b; alg = "exact") / sqrt(inner(a, a; alg = "exact") * inner(b, b; alg = "exact"))
    @test f * conj(f) ≈ 1.0 atol = 10*eps(real(elt))

    #One-site apply (no normalization)
    a = random_tensornetwork(rng, elt, s; link_space = 2)
    H = ITensorNetwork(ttn(heisenberg(g), s))
    Ha = apply(H, a; maxdim_init = 4, solver_kwargs = (; niters = 20, nsites = 1, tolerance = 1e-8, normalize = false))
    @test  inner(Ha, a; alg = "exact") / inner(a, H, a; alg = "exact") ≈ 1.0 atol = 10*eps(real(elt))

    #Two-site apply (no normalization)
    a = random_tensornetwork(rng, elt, s; link_space = 2)
    H = ITensorNetwork(ttn(heisenberg(g), s))
    Ha = apply(H, a; maxdim_init = 1, solver_kwargs= (; maxdim = 4, cutoff = 1e-16, nsites = 2, tolerance = 1e-8, normalize = false))
    @test  inner(Ha, a; alg = "exact") / inner(a, H, a; alg = "exact") ≈ 1.0 atol = 10*eps(real(elt))

    end
end

end
