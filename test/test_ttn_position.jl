@eval module $(gensym())
using Dictionaries: Dictionary, Indices
using Graphs: vertices
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using ITensorNetworks: ITensorNetwork, ProjTTN, environments, position, siteinds, ttn
using ITensors: ITensors, ITensor
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_path_graph
using NamedGraphs: NamedEdge
using Test: @test, @testset

@testset "ProjTTN position" begin
    # make a nontrivial TTN state and TTN operator

    auto_fermion_enabled = ITensors.using_auto_fermion()
    use_qns = true
    cutoff = 1.0e-12

    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    if use_qns  # test whether autofermion breaks things when using non-fermionic QNs
        ITensors.enable_auto_fermion()
    else        # when using no QNs, autofermion breaks # ToDo reference Issue in ITensors
        ITensors.disable_auto_fermion()
    end
    s = siteinds("S=1/2", c; conserve_qns = use_qns)

    os = ModelHamiltonians.heisenberg(c)

    H = ttn(os, s)

    d = Dict()
    for (i, v) in enumerate(vertices(s))
        d[v] = isodd(i) ? "Up" : "Dn"
    end
    states = v -> d[v]
    psi = ttn(states, s)

    # actual test, verifies that position is out of place
    vs = collect(vertices(s))
    PH = ProjTTN(H)
    PH = position(PH, psi, [vs[2]])
    original_keys = deepcopy(keys(environments(PH)))
    # test out-of-placeness of position
    PHc = position(PH, psi, [vs[2], vs[5]])
    @test keys(environments(PH)) == original_keys
    @test keys(environments(PHc)) != original_keys

    if !auto_fermion_enabled
        ITensors.disable_auto_fermion()
    end
end
@testset "ProjTTN construction regression test" begin
    pos = Indices{Tuple{String, Int}}()
    g = named_path_graph(2)
    operator = ttn(ITensorNetwork{Any}(g))
    environments = Dictionary{NamedEdge{Any}, ITensor}()
    @test ProjTTN(pos, operator, environments) isa ProjTTN{Any, Indices{Any}}
end
end
