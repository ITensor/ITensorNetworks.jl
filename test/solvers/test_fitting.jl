using ITensors: apply, inner
using ITensorNetworks: ITensorNetwork, siteinds, ttn, random_tensornetwork
using ITensorNetworks.ModelHamiltonians: heisenberg
using NamedGraphs.NamedGraphGenerators: named_comb_tree
using Test: @test, @testset
using Printf
using StableRNGs: StableRNG
using TensorOperations: TensorOperations #For contraction order finding

@testset "Fitting Tests" begin
    outputlevel = 1
    for elt in (Float32, Float64, Complex{Float32}, Complex{Float64})
        (outputlevel >= 1) && println("\nFitting tests with elt = ", elt)
        g = named_comb_tree((3, 2))
        s = siteinds("S=1/2", g)

        rng = StableRNG(1234)

        ##One-site truncation
        #a = random_tensornetwork(rng, elt, s; link_space=3)
        #b = truncate(a; maxdim=3)
        #f =
        #  inner(a, b; alg="exact") /
        #  sqrt(inner(a, a; alg="exact") * inner(b, b; alg="exact"))
        #(outputlevel >= 1) && @printf("One-site truncation. Fidelity = %s\n", f)
        #@test abs(abs(f) - 1.0) <= 10*eps(real(elt))

        ##Two-site truncation
        #a = random_tensornetwork(rng, elt, s; link_space=3)
        #b = truncate(a; maxdim=3, cutoff=1e-16, nsites=2)
        #f =
        #  inner(a, b; alg="exact") /
        #  sqrt(inner(a, a; alg="exact") * inner(b, b; alg="exact"))
        #(outputlevel >= 1) && @printf("Two-site truncation. Fidelity = %s\n", f)
        #@test abs(abs(f) - 1.0) <= 10*eps(real(elt))

        # #One-site apply (no normalization)
        a = random_tensornetwork(rng, elt, s; link_space = 2)
        H = ITensorNetwork(ttn(heisenberg(g), s))
        Ha = apply(H, a; maxdim = 4, nsites = 1, normalize = false)
        f = inner(Ha, a; alg = "exact") / inner(a, H, a; alg = "exact")
        (outputlevel >= 1) && @printf("One-site apply. Fidelity = %s\n", f)
        @test abs(f - 1.0) <= 500 * eps(real(elt))

        # #Two-site apply (no normalization)
        a = random_tensornetwork(rng, elt, s; link_space = 2)
        H = ITensorNetwork(ttn(heisenberg(g), s))
        Ha = apply(H, a; maxdim = 4, cutoff = 1.0e-16, nsites = 2, normalize = false)
        f = inner(Ha, a; alg = "exact") / inner(a, H, a; alg = "exact")
        (outputlevel >= 1) && @printf("Two-site apply. Fidelity = %s\n", f)
        @test abs(f - 1.0) <= 500 * eps(real(elt))
    end
end
