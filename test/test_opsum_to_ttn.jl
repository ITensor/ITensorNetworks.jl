@eval module $(gensym())
using DataGraphs: vertex_data
using Dictionaries: Dictionary
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
using ITensors.ITensorMPS: ITensorMPS
using ITensors.NDTensors: matrix
using ITensorGaussianMPS: ITensorGaussianMPS
using ITensorNetworks: ITensorNetworks, OpSum, ttn, relabel_sites, siteinds
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
    ITensors.disable_auto_fermion() # ToDo: remove when autofermion incompatibility with no QNs is fixed
    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)

    is = siteinds("S=1/2", c)

    # linearized version
    linear_order = [4, 1, 2, 5, 3, 6]
    vmap = Dictionary(vertices(is)[linear_order], 1:length(linear_order))
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

    @testset "Svd approach" for root_vertex in leaf_vertices(is)
      # get TTN Hamiltonian directly
      Hsvd = ttn(H, is; root_vertex=root_vertex, cutoff=1e-10)
      # get corresponding MPO Hamiltonian
      Hline = ITensorMPS.MPO(relabel_sites(H, vmap), sites)
      # compare resulting dense Hamiltonians
      @disable_warn_order begin
        Tttno = prod(Hline)
        Tmpo = contract(Hsvd)
      end
      @test Tttno ≈ Tmpo rtol = 1e-6

      # this breaks for longer range interactions
      Hsvd_lr = ttn(Hlr, is; root_vertex=root_vertex, algorithm="svd", cutoff=1e-10)
      Hline_lr = ITensorMPS.MPO(relabel_sites(Hlr, vmap), sites)
      @disable_warn_order begin
        Tttno_lr = prod(Hline_lr)
        Tmpo_lr = contract(Hsvd_lr)
      end
      @test Tttno_lr ≈ Tmpo_lr rtol = 1e-6
    end
    if auto_fermion_enabled
      ITensors.enable_auto_fermion()
    end
  end

  @testset "Multiple onsite terms (regression test for issue #62)" begin
    auto_fermion_enabled = ITensors.using_auto_fermion()
    ITensors.disable_auto_fermion() # ToDo: remove when autofermion incompatibility with no QNs is fixed
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
    vmap = Dictionary(vertices(is)[linear_order], 1:length(linear_order))
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

    @testset "Svd approach" for root_vertex in leaf_vertices(is)
      # get TTN Hamiltonian directly
      Hsvd = ttn(H, is; root_vertex=root_vertex, cutoff=1e-10)
      # get corresponding MPO Hamiltonian
      Hline = ITensorMPS.MPO(relabel_sites(H, vmap), sites)
      # compare resulting sparse Hamiltonians

      @disable_warn_order begin
        Tmpo = prod(Hline)
        Tttno = contract(Hsvd)
      end
      @test Tttno ≈ Tmpo rtol = 1e-6

      # this breaks for longer range interactions ###not anymore
      Hsvd_lr = ttn(Hlr, is; root_vertex=root_vertex, algorithm="svd", cutoff=1e-10)
      Hline_lr = ITensorMPS.MPO(relabel_sites(Hlr, vmap), sites)
      @disable_warn_order begin
        Tttno_lr = prod(Hline_lr)
        Tmpo_lr = contract(Hsvd_lr)
      end
      @test Tttno_lr ≈ Tmpo_lr rtol = 1e-6
    end
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

    @testset "Svd approach" for root_vertex in leaf_vertices(is)
      # get TTN Hamiltonian directly
      Hsvd = ttn(H, is; root_vertex=root_vertex, cutoff=1e-10)
      # get corresponding MPO Hamiltonian
      sites = [only(is[v]) for v in reverse(post_order_dfs_vertices(c, root_vertex))]
      vmap = Dictionary(reverse(post_order_dfs_vertices(c, root_vertex)), 1:length(sites))
      Hline = ITensorMPS.MPO(relabel_sites(H, vmap), sites)
      # compare resulting sparse Hamiltonians
      Hmat_sp = ITensorGaussianMPS.hopping_hamiltonian(relabel_sites(H, vmap))
      @disable_warn_order begin
        Tmpo = prod(Hline)
        Tttno = contract(Hsvd)
      end

      # verify that the norm isn't 0 and thus the same (which would indicate a problem with the autofermion system
      @test norm(Tmpo) > 0
      @test norm(Tttno) > 0
      @test norm(Tmpo) ≈ norm(Tttno) rtol = 1e-6

      @test_broken Tmpo ≈ Tttno # ToDo fix comparison for fermionic tensors
      # In the meantime: matricize tensors and convert to dense Matrix to compare element by element
      dTmm = to_matrix(Tmpo)
      dTtm = to_matrix(Tttno)
      @test any(>(1e-14), dTmm - dTtm)

      # also compare with energies obtained from single-particle Hamiltonian
      GS_mb, _, _ = eigsolve(dTtm, 1, :SR, eltype(dTtm))
      spectrum_sp = eigvals(Hmat_sp)
      @test minimum(cumsum(spectrum_sp)) ≈ GS_mb[1] atol = 1e-8
    end
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
    vmap = Dictionary(vertices(is)[linear_order], 1:length(linear_order))
    sites = only.(filter(d -> !isempty(d), collect(vertex_data(is_missing_site))))[linear_order]

    J1 = -1
    J2 = 2
    h = 0.5
    # connectivity of the Hamiltonian is that of the original comb graph
    H = ModelHamiltonians.heisenberg(c; J1=J1, J2=J2, h=h)

    # add combination of longer range interactions
    Hlr = copy(H)
    Hlr += 5, "Z", (1, 2), "Z", (2, 2)
    Hlr += -4, "Z", (1, 1), "Z", (2, 2)
    Hlr += 2.0, "Z", (2, 2), "Z", (3, 2)
    Hlr += -1.0, "Z", (1, 2), "Z", (3, 1)

    @testset "Svd approach" for root_vertex in leaf_vertices(is)
      # get TTN Hamiltonian directly
      Hsvd = ttn(H, is_missing_site; root_vertex=root_vertex, cutoff=1e-10)
      # get corresponding MPO Hamiltonian
      Hline = ITensorMPS.MPO(relabel_sites(H, vmap), sites)

      # compare resulting sparse Hamiltonians
      @disable_warn_order begin
        Tmpo = prod(Hline)
        Tttno = contract(Hsvd)
      end
      @test Tttno ≈ Tmpo rtol = 1e-6

      Hsvd_lr = ttn(
        Hlr, is_missing_site; root_vertex=root_vertex, algorithm="svd", cutoff=1e-10
      )
      Hline_lr = ITensorMPS.MPO(relabel_sites(Hlr, vmap), sites)
      @disable_warn_order begin
        Tttno_lr = prod(Hline_lr)
        Tmpo_lr = contract(Hsvd_lr)
      end
      @test Tttno_lr ≈ Tmpo_lr rtol = 1e-6
    end
  end
end
end
