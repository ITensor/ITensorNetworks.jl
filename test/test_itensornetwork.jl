@eval module $(gensym())
using Dictionaries: Dictionary
using Distributions: Uniform
using Graphs:
  degree,
  dijkstra_shortest_paths,
  edges,
  grid,
  has_vertex,
  ne,
  neighbors,
  nv,
  rem_vertex!,
  vertices,
  weights
using GraphsFlows: GraphsFlows
using ITensors:
  ITensors,
  Index,
  ITensor,
  Op,
  commonind,
  commoninds,
  contract,
  dag,
  hascommoninds,
  hasinds,
  inds,
  inner,
  itensor,
  onehot,
  order,
  random_itensor,
  scalartype,
  sim,
  uniqueinds
using ITensors.NDTensors: NDTensors, dim
using ITensorNetworks:
  ITensorNetworks,
  ⊗,
  IndsNetwork,
  ITensorNetwork,
  contraction_sequence,
  flatten_linkinds,
  flatten_siteinds,
  inner_network,
  linkinds,
  neighbor_tensors,
  norm_sqr,
  norm_sqr_network,
  orthogonalize,
  random_tensornetwork,
  siteinds,
  ttn
using LinearAlgebra: factorize
using NamedGraphs: NamedEdge
using NamedGraphs.GraphsExtensions: incident_edges
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using NDTensors: NDTensors, dim
using Random: randn!
using StableRNGs: StableRNG
using Test: @test, @test_broken, @testset
const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "ITensorNetwork tests" begin
  @testset "ITensorNetwork Basics" begin
    g = named_grid((4,))
    s = siteinds("S=1/2", g)
    @test s isa IndsNetwork
    @test nv(s) == 4
    @test ne(s) == 3
    @test neighbors(s, (2,)) == [(1,), (3,)]
    tn = ITensorNetwork(s; link_space=2)
    @test nv(tn) == 4
    @test ne(tn) == 3
    @test tn isa ITensorNetwork
    @test neighbors(tn, (2,)) == [(1,), (3,)]

    # TODO: How to support this syntax?
    @test_broken tn[1] isa ITensor
    @test tn[(1,)] isa ITensor

    # TODO: How to support this syntax?
    @test_broken order(tn[1]) == 2
    @test order(tn[(1,)]) == 2

    # TODO: How to support this syntax?
    @test_broken tn[2] isa ITensor
    @test tn[(2,)] isa ITensor

    # TODO: How to support this syntax?
    @test_broken order(tn[2]) == 3
    @test order(tn[(2,)]) == 3

    # XXX: Slicing syntax is no long supported, use `induced_subgraph`.
    @test_broken tn[1:2] isa ITensorNetwork
    # TODO: Support this syntax, maybe rename `subgraph`.
    @test_broken induced_subgraph(tn, [(1,), (2,)]) isa ITensorNetwork
    rng = StableRNG(1234)
    for v in vertices(tn)
      tn[v] = randn!(rng, tn[v])
    end
    tn′ = sim(dag(tn); sites=[])

    @test tn′ isa ITensorNetwork
    inner_tn = tn ⊗ tn′
    @test inner_tn isa ITensorNetwork
    sequence = contraction_sequence(inner_tn)
    @test sequence isa Vector
    inner_res = contract(inner_tn; sequence)[]
    @test inner_res isa Float64

    # test that by default vertices are linked by bond-dimension 1 index
    tn = ITensorNetwork(s)
    @test isone(ITensors.dim(commonind(tn[(1,)], tn[(2,)])))
  end

  @testset "Constructors from ITensors" begin
    i, j, k, l = Index.(fill(2, 4))
    A = ITensor(i, j)
    B = ITensor(j, k)
    C = ITensor(k, l)

    tn = ITensorNetwork([A, B, C])
    @test issetequal(vertices(tn), [1, 2, 3])
    @test issetequal(edges(tn), NamedEdge.([1 => 2, 2 => 3]))

    tn = ITensorNetwork(["A", "B", "C"], [A, B, C])
    @test issetequal(vertices(tn), ["A", "B", "C"])
    @test issetequal(edges(tn), NamedEdge.(["A" => "B", "B" => "C"]))

    tn = ITensorNetwork(["A" => A, "B" => B, "C" => C])
    @test issetequal(vertices(tn), ["A", "B", "C"])
    @test issetequal(edges(tn), NamedEdge.(["A" => "B", "B" => "C"]))
  end

  @testset "Contract edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(v -> "↑", s)
    tn = norm_sqr_network(ψ)
    tn_2 = contract(tn, ((1, 2), "ket") => ((1, 2), "bra"))
    @test !has_vertex(tn_2, ((1, 2), "ket"))
    @test tn_2[((1, 2), "bra")] ≈ tn[((1, 2), "ket")] * tn[((1, 2), "bra")]
  end

  @testset "Remove edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(v -> "↑", s)
    rem_vertex!(ψ, (1, 2))
    tn = norm_sqr_network(ψ)
    @test !has_vertex(tn, ((1, 2), "bra"))
    @test !has_vertex(tn, ((1, 2), "ket"))
    @test has_vertex(tn, ((1, 1), "bra"))
    @test has_vertex(tn, ((1, 1), "ket"))
    @test has_vertex(tn, ((2, 1), "bra"))
    @test has_vertex(tn, ((2, 1), "ket"))
    @test has_vertex(tn, ((2, 2), "bra"))
    @test has_vertex(tn, ((2, 2), "ket"))
  end

  @testset "Custom element type (eltype=$elt)" for elt in elts,
    kwargs in ((;), (; link_space=3)),
    g in (
      grid((4,)),
      named_grid((3, 3)),
      siteinds("S=1/2", grid((4,))),
      siteinds("S=1/2", named_grid((3, 3))),
    )

    rng = StableRNG(1234)
    ψ = ITensorNetwork(g; kwargs...) do v
      return inds -> itensor(randn(rng, elt, dim.(inds)...), inds)
    end
    @test eltype(ψ[first(vertices(ψ))]) == elt

    ψc = conj(ψ)
    for v in vertices(ψ)
      @test ψc[v] == conj(ψ[v])
    end

    ψd = dag(ψ)
    for v in vertices(ψ)
      @test ψd[v] == dag(ψ[v])
    end

    rng = StableRNG(1234)
    ψ = ITensorNetwork(g; kwargs...) do v
      return inds -> itensor(randn(rng, dim.(inds)...), inds)
    end
    @test eltype(ψ[first(vertices(ψ))]) == Float64
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, elt, g; kwargs...)
    @test eltype(ψ[first(vertices(ψ))]) == elt
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, g; kwargs...)
    @test eltype(ψ[first(vertices(ψ))]) == Float64
    ψ = ITensorNetwork(elt, undef, g; kwargs...)
    @test eltype(ψ[first(vertices(ψ))]) == elt
    ψ = ITensorNetwork(undef, g)
    @test eltype(ψ[first(vertices(ψ))]) == Float64
  end

  @testset "Product state constructors" for elt in elts
    dims = (2, 2)
    g = named_comb_tree(dims)
    s = siteinds("S=1/2", g)
    state1 = ["↑" "↓"; "↓" "↑"]
    state2 = reshape([[1, 0], [0, 1], [0, 1], [1, 0]], 2, 2)
    each_args = (;
      ferro=(
        ("↑",),
        (elt, "↑"),
        (Returns(i -> ITensor([1, 0], i)),),
        (elt, Returns(i -> ITensor([1, 0], i))),
        (Returns([1, 0]),),
        (elt, Returns([1, 0])),
      ),
      antiferro=(
        (state1,),
        (elt, state1),
        (Dict(CartesianIndices(dims) .=> state1),),
        (elt, Dict(CartesianIndices(dims) .=> state1)),
        (Dict(Tuple.(CartesianIndices(dims)) .=> state1),),
        (elt, Dict(Tuple.(CartesianIndices(dims)) .=> state1)),
        (Dictionary(CartesianIndices(dims), state1),),
        (elt, Dictionary(CartesianIndices(dims), state1)),
        (Dictionary(Tuple.(CartesianIndices(dims)), state1),),
        (elt, Dictionary(Tuple.(CartesianIndices(dims)), state1)),
        (state2,),
        (elt, state2),
        (Dict(CartesianIndices(dims) .=> state2),),
        (elt, Dict(CartesianIndices(dims) .=> state2)),
        (Dict(Tuple.(CartesianIndices(dims)) .=> state2),),
        (elt, Dict(Tuple.(CartesianIndices(dims)) .=> state2)),
        (Dictionary(CartesianIndices(dims), state2),),
        (elt, Dictionary(CartesianIndices(dims), state2)),
        (Dictionary(Tuple.(CartesianIndices(dims)), state2),),
        (elt, Dictionary(Tuple.(CartesianIndices(dims)), state2)),
      ),
    )
    for pattern in keys(each_args)
      for args in each_args[pattern]
        x = ITensorNetwork(args..., s)
        if first(args) === elt
          @test scalartype(x) === elt
        else
          @test scalartype(x) === Float64
        end
        for v in vertices(x)
          xᵛ = x[v]
          @test degree(x, v) + 1 == ndims(xᵛ)
          sᵛ = only(siteinds(x, v))
          for w in neighbors(x, v)
            lʷ = only(linkinds(x, v => w))
            @test dim(lʷ) == 1
            xᵛ *= onehot(lʷ => 1)
          end
          @test ndims(xᵛ) == 1
          a = if pattern == :ferro
            [1, 0]
          elseif pattern == :antiferro
            iseven(sum(v)) ? [1, 0] : [0, 1]
          end
          @test xᵛ == ITensor(a, sᵛ)
        end
      end
    end
  end
  @testset "random_tensornetwork with custom distributions" begin
    distribution = Uniform(-1.0, 1.0)
    rng = StableRNG(1234)
    tn = random_tensornetwork(rng, distribution, named_grid(4); link_space=2)
    # Note: distributions in package `Distributions` currently doesn't support customized
    # eltype, and all elements have type `Float64`
    @test eltype(tn[first(vertices(tn))]) == Float64
  end
  @testset "orthogonalize" begin
    rng = StableRNG(1234)
    tn = random_tensornetwork(rng, named_grid(4); link_space=2)
    Z = norm_sqr(tn)
    tn_ortho = factorize(tn, 4 => 3)
    # TODO: Error here in arranging the edges. Arrange by hash?
    Z̃ = norm_sqr(tn_ortho)
    @test nv(tn_ortho) == 5
    @test nv(tn) == 4
    @test Z ≈ Z̃
    tn_ortho = orthogonalize(tn, 4 => 3)
    Z̃ = norm_sqr(tn_ortho)
    @test nv(tn_ortho) == 4
    @test nv(tn) == 4
    @test Z ≈ Z̃

    tn_ortho = orthogonalize(tn, 1)
    Z̃ = norm_sqr(tn_ortho)
    @test Z ≈ Z̃
    Z̃ = inner(tn_ortho, tn)
    @test Z ≈ Z̃
  end

  @testset "dijkstra_shortest_paths" begin
    tn = ITensorNetwork(named_grid(4); link_space=2)
    paths = dijkstra_shortest_paths(tn, [1])
    @test paths.dists == Dictionary([0, 1, 2, 3])
    @test paths.parents == Dictionary([1, 1, 2, 3])
    @test paths.pathcounts == Dictionary([1.0, 1.0, 1.0, 1.0])
  end

  @testset "mincut" begin
    tn = ITensorNetwork(named_grid(4); link_space=3)
    w = weights(tn)
    @test w isa Dictionary{Tuple{Int,Int},Float64}
    @test length(w) ≈ ne(tn)
    @test w[(1, 2)] ≈ log2(3)
    @test w[(2, 3)] ≈ log2(3)
    @test w[(3, 4)] ≈ log2(3)
    p1, p2, wc = GraphsFlows.mincut(tn, 2, 3)
    @test issetequal(p1, [1, 2])
    @test issetequal(p2, [3, 4])
    @test isone(wc)

    p1, p2, wc = GraphsFlows.mincut(tn, 2, 3, w)
    @test issetequal(p1, [1, 2])
    @test issetequal(p2, [3, 4])
    @test wc ≈ log2(3)
  end

  @testset "Index access" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s; link_space=2)

    nt = neighbor_tensors(ψ, (1, 1))
    @test length(nt) == 2
    @test all(map(hascommoninds(ψ[1, 1]), nt))

    @test all(map(t -> isempty(commoninds(inds(t), uniqueinds(ψ, (1, 1)))), nt))

    e = (1, 1) => (2, 1)
    uie = uniqueinds(ψ, e)
    @test isempty(commoninds(uie, inds(ψ[2, 1])))
    @test issetequal(uie, union(commoninds(ψ[1, 1], ψ[1, 2]), uniqueinds(ψ, (1, 1))))

    @test siteinds(ψ, (1, 1)) == s[1, 1]

    cie = commoninds(ψ, e)
    @test hasinds(ψ[1, 1], cie) && hasinds(ψ[2, 1], cie)
    @test isempty(commoninds(uie, cie))

    @test linkinds(ψ, e) == commoninds(ψ[1, 1], ψ[2, 1])

    @test length(flatten_siteinds(ψ)) == length(vertices(g))
    @test length(flatten_linkinds(ψ)) == length(edges(g))
  end

  @testset "eltype conversion, $new_eltype" for new_eltype in (Float32, ComplexF64)
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    rng = StableRNG(1234)
    ψ = random_tensornetwork(rng, s; link_space=2)
    @test scalartype(ψ) == Float64
    ϕ = NDTensors.convert_scalartype(new_eltype, ψ)
    @test scalartype(ϕ) == new_eltype
  end

  @testset "Construction from state map" for elt in (Float32, ComplexF64)
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    state_map(v::Tuple) = iseven(sum(isodd.(v))) ? "↑" : "↓"
    ψ = ITensorNetwork(state_map, s)
    t = ψ[2, 2]
    si = only(siteinds(ψ, (2, 2)))
    bi = map(e -> only(linkinds(ψ, e)), incident_edges(ψ, (2, 2)))
    @test eltype(t) == Float64
    @test abs(t[si => "↑", [b => end for b in bi]...]) == 1.0 # insert_links introduces extra signs through factorization...
    @test t[si => "↓", [b => end for b in bi]...] == 0.0
    ϕ = ITensorNetwork(elt, state_map, s)
    t = ϕ[2, 2]
    si = only(siteinds(ϕ, (2, 2)))
    bi = map(e -> only(linkinds(ϕ, e)), incident_edges(ϕ, (2, 2)))
    @test eltype(t) == elt
    @test abs(t[si => "↑", [b => end for b in bi]...]) == convert(elt, 1.0) # insert_links introduces extra signs through factorization...
    @test t[si => "↓", [b => end for b in bi]...] == convert(elt, 0.0)
  end

  @testset "Priming and tagging" begin
    # TODO: add actual tests
    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    is = siteinds("S=1/2", c)
    rng = StableRNG(1234)
    tn = random_tensornetwork(rng, is; link_space=3)
    @test_broken swapprime(tn, 0, 2)
  end
end
end
