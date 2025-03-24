@eval module $(gensym())
using ITensorNetworks: random_tensornetwork, siteinds, BilinearFormNetwork, subgraph, inner, truncate, maximize_bilinearform, union_all_inds
using Graphs: vertices
using NamedGraphs.NamedGraphGenerators: named_grid, named_comb_tree
using SplitApplyCombine: group
using StableRNGs: StableRNG
using TensorOperations: TensorOperations
using Test: @test, @test_broken, @testset
using ITensors: apply, dag, prime


@testset "TNS Fitting" for elt in (
    Float32, Float64, Complex{Float32}, Complex{Float64}
  )
    begin

    rng = StableRNG(1234)

    g = named_comb_tree((3,2))
    s = siteinds("S=1/2", g)

    a = random_tensornetwork(rng, elt, s; link_space = 3)
    b = truncate(a; maxdim = 3)
    f = inner(a, b; alg = "exact") / sqrt(inner(a, a; alg = "exact") * inner(b, b; alg = "exact"))
    @show f * conj(f)
    #@test f * conj(f) ≈ 1.0 atol = 10*eps(real(elt))

    H = random_tensornetwork(elt, union_all_inds(s, prime(s)); link_space = 3)

    Ha1 = apply(H, a; maxdim = 6)
    Ha2 = apply(H, a; maxdim = 9)

    f = inner(Ha1, Ha2; alg = "exact") / sqrt(inner(Ha1, Ha1; alg = "exact") * inner(Ha2, Ha2; alg = "exact"))
    @show f * conj(f)
    
    end
end

end
