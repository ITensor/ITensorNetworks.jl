@eval module $(gensym())
using ITensorNetworks: ITensorNetworks
using Suppressor: @suppress
using Test: @testset

@testset "Test examples" begin
  example_files = [
    "README.jl",
    "boundary.jl",
    "distances.jl",
    "examples.jl",
    "mincut.jl",
    "mps.jl",
    "peps.jl",
    "steiner_tree.jl",
    "dynamics/2d_ising_imag_tebd.jl",
    "treetensornetworks/comb_tree.jl",
    "treetensornetworks/spanning_tree.jl",
    "treetensornetworks/ttn_basics.jl",
    "treetensornetworks/ttn_type.jl",
  ]
  @testset "Test $example_file" for example_file in example_files
    @suppress include(joinpath(pkgdir(ITensorNetworks), "examples", example_file))
  end
  if !Sys.iswindows()
    example_files = ["contraction_sequence/contraction_sequence.jl"]
    @testset "Test $example_file (using KaHyPar, so no Windows support)" for example_file in
                                                                             example_files
      @suppress include(joinpath(pkgdir(ITensorNetworks), "examples", example_file))
    end
  end
end
end
