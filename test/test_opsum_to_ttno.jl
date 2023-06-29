using Dictionaries
using ITensors
using ITensorNetworks
using Random
using Test

@testset "OpSum to TTN converter" begin
  @testset "OpSum to TTN" begin
    # small comb tree
    tooth_lengths = fill(2, 3)
    c = named_comb_tree(tooth_lengths)
    root_vertex = (3, 2)
    is = siteinds("S=1/2", c)

    # linearized version
    linear_order = [4, 1, 2, 5, 3, 6]
    vmap = Dictionary(vertices(is)[linear_order], 1:length(linear_order))
    sites = only.(collect(vertex_data(is)))[linear_order]

    # test with next-to-nearest-neighbor Ising Hamiltonian
    J1 = -1
    J2 = 2
    h = 0.5
    H = ising(c; J1=J1, J2=J2, h=h)

    # add combination of longer range interactions
    Hlr = copy(H)
    Hlr += 5, "Z", (1, 2), "Z", (2, 2)
    Hlr += -4, "Z", (1, 1), "Z", (2, 2)

    @testset "Finite state machine" begin
      # get TTN Hamiltonian directly
      Hfsm = TTN(H, is; root_vertex=root_vertex, method=:fsm, cutoff=1e-10)
      # get corresponding MPO Hamiltonian
      Hline = MPO(relabel_sites(H, vmap), sites)
      # compare resulting dense Hamiltonians
      @disable_warn_order begin
        Tttno = prod(Hline)
        Tmpo = contract(Hfsm)
      end
      @test Tttno ≈ Tmpo rtol = 1e-6

      # same thing for longer range interactions
      Hfsm_lr = TTN(Hlr, is; root_vertex=root_vertex, method=:fsm, cutoff=1e-10)
      Hline_lr = MPO(relabel_sites(Hlr, vmap), sites)
      @disable_warn_order begin
        Tttno_lr = prod(Hline_lr)
        Tmpo_lr = contract(Hfsm_lr)
      end
      @test Tttno_lr ≈ Tmpo_lr rtol = 1e-6
    end

    @testset "Svd approach" begin
      # get TTN Hamiltonian directly
      Hsvd = TTN(H, is; root_vertex=root_vertex, method=:svd, cutoff=1e-10)
      # get corresponding MPO Hamiltonian
      Hline = MPO(relabel_sites(H, vmap), sites)
      # compare resulting dense Hamiltonians
      @disable_warn_order begin
        Tttno = prod(Hline)
        Tmpo = contract(Hsvd)
      end
      @test Tttno ≈ Tmpo rtol = 1e-6

      # this breaks for longer range interactions
      Hsvd_lr = TTN(Hlr, is; root_vertex=root_vertex, method=:svd, cutoff=1e-10)
      Hline_lr = MPO(relabel_sites(Hlr, vmap), sites)
      @disable_warn_order begin
        Tttno_lr = prod(Hline_lr)
        Tmpo_lr = contract(Hsvd_lr)
      end
      @test_broken Tttno_lr ≈ Tmpo_lr rtol = 1e-6
    end
  end

  @testset "Multiple onsite terms (regression test for issue #62)" begin
    grid_dims = (2, 1)
    g = named_grid(grid_dims)
    s = siteinds("S=1/2", g)

    os1 = OpSum()
    os1 += 1.0, "Sx", (1, 1)
    os2 = OpSum()
    os2 += 1.0, "Sy", (1, 1)
    H1 = TTN(os1, s)
    H2 = TTN(os2, s)
    H3 = TTN(os1 + os2, s)

    @test H1 + H2 ≈ H3 rtol = 1e-6
  end
end
