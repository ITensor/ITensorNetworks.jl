using ITensorNetworks
using Test

@testset "ApproximateTNContraction.jl" begin
  for filename in [
    "lattice.jl",
    "models.jl",
    "tree.jl",
    "indexgroup.jl",
    "cache.jl",
    "treetensor.jl",
    # "interface.jl",
  ]
    println("Running $filename in ApproximateTNContraction.jl")
    include(filename)
  end
end
