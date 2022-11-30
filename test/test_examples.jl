using ITensorNetworks
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
    joinpath("ttn", "comb_tree.jl"),
    joinpath("ttn", "spanning_tree.jl"),
    joinpath("ttn", "ttn_basics.jl"),
    joinpath("ttn", "ttn_type.jl"),
  ]
  @testset "Test $example_file" for example_file in example_files
    include(joinpath(pkgdir(ITensorNetworks), "examples", example_file))
  end
end
