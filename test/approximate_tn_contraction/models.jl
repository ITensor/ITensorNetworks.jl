using ITensors
using ITensorNetworks
using ITensorNetworks.ApproximateTNContraction: Models

@testset "test local hamiltonian builder" begin
  Nx = 2
  Ny = 3
  sites = siteinds("S=1/2", Ny, Nx)
  H = Models.mpo(Models.Model("tfim"), sites; h=1.0)
  H_local = Models.localham(Models.Model("tfim"), sites; h=1.0)
  H_line = Models.lineham(Models.Model("tfim"), sites; h=1.0)
  @test Models.checkham(H_local, H, sites)
  @test Models.checkham(H_line, H, sites)
end
