using ITensors
using ITensorNetworks.ApproximateTNContraction: IndexGroup, get_igs_cache_info

@testset "test get_igs_cache_info" begin
  i = IndexGroup([Index(2, "i")])
  j = IndexGroup([Index(3, "j")])
  k = IndexGroup([Index(2, "k")])
  l = IndexGroup([Index(4, "l")])
  m = IndexGroup([Index(5, "m")])
  n = IndexGroup([Index(5, "n")])

  type = Vector{IndexGroup}
  igs_list = [type([l, k, i, m, n]), type([i, j, k]), type([l, j, m, n])]
  contract_igs_list = [type([m]), type([j]), type([j])]
  out = get_igs_cache_info(igs_list, contract_igs_list)
  @test out == ([l, k, i, m, n], [], [])
end
