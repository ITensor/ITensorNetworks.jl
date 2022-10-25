using ITensors
using ITensorNetworks
using ITensorNetworks.ApproximateTNContraction: Square, bonds

@testset "test lattice" begin
  lattice = Square((2, 3))
  bds = bonds(lattice; periodic=false)
  @test length(bds) == 7
end
