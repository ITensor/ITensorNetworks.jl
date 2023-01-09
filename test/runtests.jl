using Test
using ITensorNetworks

@testset "ITensorNetworks.jl, test directory $dirname" for dirname in [
    ".",
    joinpath("test_treetensornetwork", "test_solvers"),
  ]
  test_path = joinpath(pkgdir(ITensorNetworks), "test", dirname)
  test_files = filter(readdir(test_path)) do file
    startswith(file, "test_") && endswith(file, ".jl")
  end
  @testset "Test file $filename" for filename in test_files
    include(joinpath(test_path, dirname, filename))
  end
end

nothing
