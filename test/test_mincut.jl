using ITensors, ITensorNetworks

@testset "test mincut functions on top of MPS" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  n = Index(2, "n")
  o = Index(2, "o")
  p = Index(2, "p")

  T = randomITensor(i, j, k, l, m, n, o, p)
  M = MPS(T, (i, j, k, l, m, n, o, p); cutoff=1e-5, maxdim=500)
  network = M[:]
  out = inds_binary_tree(network, [i, j, k, l, m, n, o, p]; algorithm="mincut")
  @test length(out) == 2
  out = inds_binary_tree(network, [i, j, k, l, m, n, o, p]; algorithm="mps")
  @test length(out) == 2
  out = inds_mps_order(network, [o, p, i, j, k, l, m, n])
  @test out in [[i, j, k, l, m, n, o, p], [p, o, n, m, l, k, j, i]]
  p1, p2 = mincut_partitions(network, [k, l], [m, n])
  # When MPS bond dimensions are large, the partition will not across internal inds
  @test (length(p1) == 0) || (length(p2) == 0)

  M = MPS(T, (i, j, k, l, m, n, o, p); cutoff=1e-5, maxdim=2)
  network = M[:]
  p1, p2 = mincut_partitions(network, [k, l], [m, n])
  # When MPS bond dimensions are small, the partition will across internal inds
  @test sort(p1) == [1, 2, 3, 4]
  @test sort(p2) == [5, 6, 7, 8]
end

@testset "test inds_binary_tree of a 2D network" begin
  N = (3, 3, 3)
  linkdim = 2
  network = randomITensorNetwork(IndsNetwork(named_grid(N)); link_space=linkdim)
  tn = Array{ITensor,length(N)}(undef, N...)
  for v in vertices(network)
    tn[v...] = network[v...]
  end
  network = vec(tn[:, :, 1])
  out = inds_binary_tree(network, noncommoninds(network...); algorithm="mincut")
  @test length(out) == 2
  out = inds_binary_tree(network, noncommoninds(network...); algorithm="mps")
  @test length(out) == 2
end
