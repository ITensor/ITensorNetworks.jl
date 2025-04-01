@eval module $(gensym())
using DataGraphs: vertex_data
using Dictionaries: Dictionary, getindices
using Graphs: add_vertex!, rem_vertex!, add_edge!, rem_edge!, vertices
using ITensors:
  ITensors,
  Index,
  ITensor,
  @disable_warn_order,
  combinedind,
  combiner,
  contract,
  dag,
  inds,
  removeqns
using ITensors.NDTensors: matrix
using ITensorGaussianMPS: ITensorGaussianMPS
using ITensorNetworks: ITensorNetworks, OpSum, ttn, siteinds
using ITensorNetworks.ITensorsExtensions: replace_vertices
using ITensorNetworks.ModelHamiltonians: ModelHamiltonians
using KrylovKit: eigsolve
using LinearAlgebra: eigvals, norm
using NamedGraphs.GraphsExtensions: leaf_vertices, post_order_dfs_vertices
using NamedGraphs.NamedGraphGenerators: named_comb_tree, named_grid
using Test: @test, @test_broken, @testset

function to_matrix(t::ITensor)
  c = combiner(inds(t; plev=0))
  tc = (t * c) * dag(c')
  cind = combinedind(c)
  return matrix(tc, cind', cind)
end

@testset "OpSum to TTN converter" begin
  @testset "OpSum to TTN" begin
    # small comb tree
    auto_fermion_enabled = ITensors.using_auto_fermion()
    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)

    is = siteinds("S=1/2", c)

    # linearized version
    linear_order = [4, 1, 2, 5, 3, 6]
    vmap = Dictionary(collect(vertices(is))[linear_order], eachindex(linear_order))
    sites = only.(collect(vertex_data(is)))[linear_order]

    # test with next-to-nearest-neighbor Ising Hamiltonian
    J1 = -1
    J2 = 2
    h = 0.5
    H = ModelHamiltonians.ising(c; J1=J1, J2=J2, h=h)
    # add combination of longer range interactions
    Hlr = copy(H)
    Hlr += 5, "Z", (1, 2), "Z", (2, 2), "Z", (3, 2)
    Hlr += -4, "Z", (1, 1), "Z", (2, 2), "Z", (3, 1)
    Hlr += 2.0, "Z", (2, 2), "Z", (3, 2)
    Hlr += -1.0, "Z", (1, 2), "Z", (3, 1)

    # root_vertex = (1, 2)
    # println(leaf_vertices(is))

    if auto_fermion_enabled
      ITensors.enable_auto_fermion()
    end
  end

  @testset "Multiple onsite terms (regression test for issue #62)" begin
    auto_fermion_enabled = ITensors.using_auto_fermion()
    if !auto_fermion_enabled
      ITensors.enable_auto_fermion()
    end
    grid_dims = (2, 1)
    g = named_grid(grid_dims)
    s = siteinds("S=1/2", g)

    os1 = OpSum()
    os1 += 1.0, "Sx", (1, 1)
    os2 = OpSum()
    os2 += 1.0, "Sy", (1, 1)
    H1 = ttn(os1, s)
    H2 = ttn(os2, s)
    H3 = ttn(os1 + os2, s)

    @test H1 + H2 ≈ H3 rtol = 1e-6
    if auto_fermion_enabled
      ITensors.enable_auto_fermion()
    end
  end

  @testset "OpSum to TTN QN" begin
    # small comb tree
    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    is = siteinds("S=1/2", c; conserve_qns=true)
    is_noqns = copy(is)
    for v in vertices(is)
      is_noqns[v] = removeqns(is_noqns[v])
    end

    # linearized version
    linear_order = [4, 1, 2, 5, 3, 6]
    vmap = Dictionary(collect(vertices(is))[linear_order], eachindex(linear_order))
    sites = only.(collect(vertex_data(is)))[linear_order]

    # test with next-to-nearest-neighbor Ising Hamiltonian
    J1 = -1
    J2 = 2
    h = 0.5
    H = ModelHamiltonians.heisenberg(c; J1=J1, J2=J2, h=h)
    # add combination of longer range interactions
    Hlr = copy(H)
    Hlr += 5, "Z", (1, 2), "Z", (2, 2)#, "Z", (3,2)
    Hlr += -4, "Z", (1, 1), "Z", (2, 2)
    Hlr += 2.0, "Z", (2, 2), "Z", (3, 2)
    Hlr += -1.0, "Z", (1, 2), "Z", (3, 1)

    # root_vertex = (1, 2)
    # println(leaf_vertices(is))
  end

  @testset "OpSum to TTN Fermions" begin
    # small comb tree
    auto_fermion_enabled = ITensors.using_auto_fermion()
    if !auto_fermion_enabled
      ITensors.enable_auto_fermion()
    end
    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    is = siteinds("Fermion", c; conserve_nf=true)

    # test with next-nearest neighbor tight-binding model
    t = 1.0
    tp = 0.4
    U = 0.0
    h = 0.5
    H = ModelHamiltonians.tight_binding(c; t, tp, h)

    # add combination of longer range interactions
    Hlr = copy(H)

    if !auto_fermion_enabled
      ITensors.disable_auto_fermion()
    end
  end

  @testset "OpSum to TTN QN missing" begin
    # small comb tree
    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    c2 = copy(c)
    ## add an internal vertex into the comb graph c2
    add_vertex!(c2, (-1, 1))
    add_edge!(c2, (-1, 1) => (2, 2))
    add_edge!(c2, (-1, 1) => (3, 1))
    add_edge!(c2, (-1, 1) => (2, 1))
    rem_edge!(c2, (2, 1) => (2, 2))
    rem_edge!(c2, (2, 1) => (3, 1))

    is = siteinds("S=1/2", c; conserve_qns=true)
    is_missing_site = siteinds("S=1/2", c2; conserve_qns=true)
    is_missing_site[(-1, 1)] = Vector{Index}[]

    # linearized version
    linear_order = [4, 1, 2, 5, 3, 6]
    vmap = Dictionary(collect(vertices(is))[linear_order], eachindex(linear_order))
    sites = only.(filter(d -> !isempty(d), collect(vertex_data(is_missing_site))))[linear_order]

    J1 = -1
    J2 = 2
    h = 0.5
    # connectivity of the Hamiltonian is that of the original comb graph
    H = ModelHamiltonians.heisenberg(c; J1, J2, h)

    # add combination of longer range interactions
    Hlr = copy(H)
    Hlr += 5, "Z", (1, 2), "Z", (2, 2)
    Hlr += -4, "Z", (1, 1), "Z", (2, 2)
    Hlr += 2.0, "Z", (2, 2), "Z", (3, 2)
    Hlr += -1.0, "Z", (1, 2), "Z", (3, 1)
  end
end
end
