@eval module $(gensym())
using ITensorNetworks: ITensorNetworks
using Suppressor: @suppress
using Test: @testset

@testset "Test examples" begin
  example_files = [
    "README.jl",
  ]
  @testset "Test $example_file" for example_file in example_files
    @suppress include(joinpath(pkgdir(ITensorNetworks), "examples", example_file))
  end
end
end
