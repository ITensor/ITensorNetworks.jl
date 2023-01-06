using ITensors, TimerOutputs
using ITensorNetworks.ApproximateTNContraction:
  get_tensors,
  OrthogonalITensor,
  tree_approximation_cache,
  inds_binary_tree,
  tree_embedding,
  approximate_contract
using ITensorNetworks.ApproximateTNContraction: timer, inds_network, Models, ising_partition

include("utils.jl")

@testset "test tree approximation" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  n = Index(2, "n")
  o = Index(2, "o")
  p = Index(2, "p")
  q = Index(2, "q")
  r = Index(2, "r")
  s = Index(2, "s")
  t = Index(2, "t")
  u = Index(2, "u")
  A = OrthogonalITensor(randomITensor(i, n))
  B = OrthogonalITensor(randomITensor(j, o))
  AB = OrthogonalITensor(randomITensor(n, o, r))
  C = OrthogonalITensor(randomITensor(k, p))
  D = OrthogonalITensor(randomITensor(l, q))
  E = OrthogonalITensor(randomITensor(m, u))
  CD = OrthogonalITensor(randomITensor(p, q, s))
  ABCD = OrthogonalITensor(randomITensor(r, s, t))
  ABCDE = OrthogonalITensor(randomITensor(t, u))
  btree = [[[[i], [j]], [[k], [l]]], [m]]
  tensors = [A, B, C, D, E, AB, CD, ABCD, ABCDE]
  embedding = Dict([
    [i] => [A],
    [j] => [B],
    [k] => [C],
    [l] => [D],
    [m] => [E],
    [[i], [j]] => [AB],
    [[k], [l]] => [CD],
    [[[i], [j]], [[k], [l]]] => [ABCD],
    [[[[i], [j]], [[k], [l]]], [m]] => [ABCDE],
  ])
  out, log_norm = tree_approximation_cache(embedding, btree)
  out[btree].tensor *= exp(log_norm)
  out = get_tensors(collect(values(out)))
  @test isapprox(contract(out...), contract(get_tensors(tensors)...))
end

@testset "test MPS times MPO" begin
  N = (5, 3)
  linkdim = 3
  cutoff = 1e-15
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  x, A = tn[:, 1], tn[:, 2]
  out_true = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=linkdim * linkdim)
  out2, log_norm = approximate_contract([A, x]; cutoff=cutoff, maxdim=linkdim * linkdim)
  tsr_true = contract(out_true...)
  tsr_nrmsquare = (tsr_true * tsr_true)[1]
  @test isapprox(tsr_true, contract(out2...) * exp(log_norm))

  maxdims = [2, 4, 6, 8]
  for dim in maxdims
    out = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=dim)
    out2, log_norm = approximate_contract([A, x]; cutoff=cutoff, maxdim=dim)
    residual1 = tsr_true - contract(out...)
    residual2 = tsr_true - contract(out2...) * exp(log_norm)
    error1 = sqrt((residual1 * residual1)[1] / tsr_nrmsquare)
    error2 = sqrt((residual2 * residual2)[1] / tsr_nrmsquare)
    print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
  end
end

@testset "test inds_binary_tree" begin
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
end

@testset "test inds_binary_tree of a 2D network" begin
  N = (3, 3, 3)
  linkdim = 2
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  network = vec(tn[:, :, 1])
  out = inds_binary_tree(network, noncommoninds(network...); algorithm="mincut")
  @test length(out) == 2
  out = inds_binary_tree(network, noncommoninds(network...); algorithm="mps")
  @test length(out) == 2
end

@testset "test tree embedding" begin
  i = Index(2, "i")
  j = Index(2, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  T = randomITensor(i, j, k, l, m)
  M = MPS(T, (i, j, k, l, m); cutoff=1e-5, maxdim=5)
  network = M[:]
  out1 = contract(network...)
  inds_btree = inds_binary_tree(network, [i, j, k, l, m]; algorithm="mincut")
  tnet_dict = tree_embedding(network, inds_btree)
  network2 = vcat(collect(values(tnet_dict))...)
  out2 = contract(network2...)
  i1 = noncommoninds(network...)
  i2 = noncommoninds(network2...)
  @test (length(i1) == length(i2))
  @test isapprox(out1, out2)
end

function benchmark_peps_contraction(tn; cutoff=1e-15, maxdim=1000)
  out = peps_contraction_mpomps(tn; cutoff=cutoff, maxdim=maxdim, snake=false)
  out2 = contract_line_group(tn; cutoff=cutoff, maxdim=maxdim)
  return out[], out2
end

@testset "test PEPS" begin
  N = (8, 8)
  linkdim = 2
  cutoff = 1e-15
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  # tn = ising_partition(N, linkdim)

  ITensors.set_warn_order(100)
  maxdim = linkdim^N[2]
  out_true, out2 = benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=maxdim)
  print(out_true, out2)
  @test abs((out_true - out2) / out_true) < 1e-3

  maxdims = [i for i in 2:16]
  for dim in maxdims
    size = dim * dim * linkdim
    out, out2 = benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim)
    error1 = abs((out - out_true) / out_true)
    error2 = abs((out2 - out_true) / out_true)
    print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
  end
end

@testset "benchmark PEPS" begin
  N = (8, 8)
  linkdim = 10
  cutoff = 1e-15
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)

  dim = 20
  # warmup
  tn = map(inds -> randomITensor(inds...), tn_inds)
  ITensors.set_warn_order(100)
  benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim)

  reset_timer!(timer)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  ITensors.set_warn_order(100)
  benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim)
  show(timer)
end
