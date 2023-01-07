using ITensorNetworks
using Test

@testset "ApproximateTNContraction.jl" begin
  for filename in ["tree.jl", "indexgroup.jl", "cache.jl", "contract.jl", "interface.jl"]
    println("Running $filename in ApproximateTNContraction.jl")
    include(filename)
  end
end
