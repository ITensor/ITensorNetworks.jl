using ITensorNetworks
using Suppressor
using Test

@testset "Test examples" begin
  example_files = [
    "README.jl",
    "examples.jl",
    "mps.jl",
    "peps.jl",
    joinpath("contraction_sequence", "contraction_sequence.jl"),
    joinpath("partition", "kahypar_vs_metis.jl"),
    joinpath("partition", "partitioning.jl"),
    joinpath("peps", "ising_tebd.jl"),
    joinpath("ttns", "comb_tree.jl"),
    joinpath("ttns", "spanning_tree.jl"),
    joinpath("ttns", "ttn_basics.jl"),
    joinpath("ttns", "ttn_type.jl"),
  ]
  @testset "Test $example_file" for example_file in example_files
    @suppress include(joinpath(pkgdir(ITensorNetworks), "examples", example_file))
  end
end
