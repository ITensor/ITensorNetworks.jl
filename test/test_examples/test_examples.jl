using ITensorNetworks
using Suppressor
using Test

@testset "Test examples" begin
  example_files = [
    "README.jl",
    "boundary.jl",
    "distances.jl",
    "examples.jl",
    "group_partition.jl",
    "mincut.jl",
    "mps.jl",
    "peps.jl",
    "steiner_tree.jl",
    joinpath("belief_propagation", "bpexample.jl"),
    joinpath("belief_propagation", "bpsequences.jl"),
    joinpath("dynamics", "2d_ising_imag_tebd.jl"),
    joinpath("dynamics", "heavy_hex_ising_real_tebd.jl"),
    joinpath("treetensornetworks", "comb_tree.jl"),
    joinpath("treetensornetworks", "spanning_tree.jl"),
    joinpath("treetensornetworks", "ttn_basics.jl"),
    joinpath("treetensornetworks", "ttn_type.jl"),
  ]
  @testset "Test $example_file" for example_file in example_files
    @suppress include(joinpath(pkgdir(ITensorNetworks), "examples", example_file))
  end
  if !Sys.iswindows()
    example_files = [
      joinpath("contraction_sequence", "contraction_sequence.jl"),
      joinpath("partition", "kahypar_vs_metis.jl"),
      joinpath("partition", "partitioning.jl"),
    ]
    @testset "Test $example_file (using KaHyPar, so no Windows support)" for example_file in
                                                                             example_files
      @suppress include(joinpath(pkgdir(ITensorNetworks), "examples", example_file))
    end
  end
end
