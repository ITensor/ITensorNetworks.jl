using ITensorNetworkAD
using AutoHOOT, ITensors, Zygote
using ITensorNetworkAD.Profiler
using ITensorNetworkAD.ITensorNetworks:
  TreeTensor,
  tree_approximation,
  tree_approximation_cache,
  inds_binary_tree,
  tree_embedding,
  approximate_contract
using ITensorNetworkAD.ITensorNetworks:
  inds_network, project_boundary, Models, ising_partition
using ITensorNetworkAD.ITensorAutoHOOT: SubNetwork, batch_tensor_contraction

const itensorah = ITensorNetworkAD.ITensorAutoHOOT

include("utils.jl")

@testset "test TreeTensor" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  l = Index(4, "l")
  m = Index(5, "m")

  A = randomITensor(i, j, k)
  B = randomITensor(k, l, m)
  C = randomITensor(i, j, l, m)
  tree_A = TreeTensor(A)
  tree_B = TreeTensor(B)
  tree_C = TreeTensor(C)

  out = A * B
  network = [tree_A, tree_B]
  nodes, dict = itensorah.generate_einsum_expr([network])
  out_list = itensorah.compute_graph(nodes, dict; cutoff=1e-15, maxdim=1000)
  @test isapprox(out, ITensor(out_list[1]))

  out = A * B * C
  out2 = contract(tree_A, tree_B, tree_C; cutoff=1e-15, maxdim=1000)
  @test isapprox(out, ITensor(out2))
end

@testset "test batch_tensor_contraction" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  l = Index(2, "l")
  m = Index(2, "m")
  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, l)
  D = randomITensor(l, m)
  E = randomITensor(m, i)

  function network(A)
    tensor_network = SubNetwork(SubNetwork(A, B, C), D, E)
    out = itensorah.batch_tensor_contraction(
      TreeTensor, [tensor_network], A; cutoff=1e-15, maxdim=1000, optimize=false
    )
    return sum(out)[]
  end
  grad_A = gradient(network, A)
  @test isapprox(grad_A[1], B * C * D * E)
end

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
  A = randomITensor(i, n)
  B = randomITensor(j, o)
  AB = randomITensor(n, o, r)
  C = randomITensor(k, p)
  D = randomITensor(l, q)
  E = randomITensor(m, u)
  CD = randomITensor(p, q, s)
  ABCD = randomITensor(r, s, t)
  ABCDE = randomITensor(t, u)
  btree = [[[[i], [j]], [[k], [l]]], [m]]
  tensors = [A, B, C, D, E, AB, CD, ABCD, ABCDE]
  out = tree_approximation(tensors, btree)
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
  out2 = tree_approximation_cache(embedding, btree)
  @test isapprox(contract(out), contract(out2))
  @test isapprox(contract(out), contract(tensors...))
end

@testset "test MPS times MPO" begin
  N = (10, 3)
  linkdim = 3
  cutoff = 1e-15
  tn_inds = inds_network(N...; linkdims=linkdim)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  state = 1
  tn = project_boundary(tn, state)
  x, A = tn[:, 1], tn[:, 2]
  out_true = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=linkdim * linkdim)
  out2 = batch_tensor_contraction(
    TreeTensor,
    [SubNetwork(SubNetwork(x), SubNetwork(A))];
    cutoff=cutoff,
    maxdim=linkdim * linkdim,
  )
  tsr_true = contract(out_true...)
  tsr_nrmsquare = (tsr_true * tsr_true)[1]
  @test isapprox(tsr_true, ITensor(out2[1]))

  maxdims = [2, 4, 6, 8]
  for dim in maxdims
    out = contract(MPO(A), MPS(x); cutoff=cutoff, maxdim=dim)
    out2 = batch_tensor_contraction(
      TreeTensor, [SubNetwork(SubNetwork(x), SubNetwork(A))]; cutoff=cutoff, maxdim=dim
    )
    residual1 = tsr_true - contract(out...)
    residual2 = tsr_true - ITensor(out2[1])
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
  out = inds_binary_tree(network, [i, j, k, l, m, n, o, p]; algorithm="mincut-mps")
  @test length(out) == 2
  out = inds_binary_tree(network, [i, j, k, l, m, n, o, p]; algorithm="mps")
  @test length(out) == 2
end

@testset "test inds_binary_tree of a 2D network" begin
  N = (8, 8, 3)
  linkdim = 2
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  network = vec(tn[:, :, 1])
  out = inds_binary_tree(network, noncommoninds(network...); algorithm="mincut")
  @test length(out) == 2
  out = inds_binary_tree(network, noncommoninds(network...); algorithm="mincut-mps")
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

function benchmark_peps_contraction(tn; cutoff=1e-15, maxdim=1000, maxsize=10^15)
  N = size(tn)
  out = peps_contraction_mpomps(tn; cutoff=cutoff, maxdim=maxdim, snake=false)
  network = SubNetwork(tn[:, 1])
  for i in 2:(N[2])
    network = SubNetwork(network, SubNetwork(tn[:, i]))
  end
  out2 = batch_tensor_contraction(
    TreeTensor, [network]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, optimize=false
  )
  return out[], ITensor(out2[1])[]
end

@testset "test PEPS" begin
  N = (8, 8) #(12, 12)
  linkdim = 2
  cutoff = 1e-15
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  # tn = ising_partition(N, linkdim)

  ITensors.set_warn_order(100)
  maxdim = linkdim^N[2]
  maxsize = maxdim * maxdim * linkdim
  out_true, out2 = benchmark_peps_contraction(
    tn; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize
  )
  print(out_true, out2)
  @test abs((out_true - out2) / out_true) < 1e-3

  maxdims = [i for i in 2:16]
  for dim in maxdims
    size = dim * dim * linkdim
    out, out2 = benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim, maxsize=size)
    error1 = abs((out - out_true) / out_true)
    error2 = abs((out2 - out_true) / out_true)
    print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
  end
end

function benchmark_3D_contraction(tn; cutoff=1e-15, maxdim=1000, maxsize=10^15)
  # TODO: this sequential MPS doesn't give the desired tree when each tn[i] is a slice of the 2D surface
  out = contract(
    tn[1]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, algorithm="sequential-mps"
  )
  for i in 2:length(tn)
    out = contract(
      out, tn[i]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, algorithm="sequential-mps"
    )
  end
  out2 = contract(tn[1]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, algorithm="mps")
  for i in 2:length(tn)
    out2 = contract(
      out2, tn[i]; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize, algorithm="mps"
    )
  end
  return out[], out2[]
end

@testset "test 3-D cube with 2D grouping" begin
  do_profile(true)
  N = (3, 3, 4) #(12, 12)
  linkdim = 2
  nrows = prod([s for s in N[1:(length(N) - 1)]])
  ncols = N[length(N)]
  maxdim = linkdim^(floor(nrows))
  cutoff = 1e-15
  # tn = ising_partition(N, linkdim)
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  # snake mapping
  for k in 1:N[3]
    for j in 1:N[2]
      if iseven(j)
        tn[:, j, k] = reverse(tn[:, j, k])
      end
    end
  end
  tn = reshape(tn, (nrows, ncols))
  tn = [TreeTensor(tn[:, i]) for i in 1:ncols]
  @info size(tn)
  ITensors.set_warn_order(100)
  maxsize = maxdim * maxdim * linkdim
  out1, out2 = benchmark_3D_contraction(tn; cutoff=cutoff, maxdim=maxdim, maxsize=maxsize)
  profile_exit()
  print(out1, out2)
  @test abs((out1 - out2) / out1) < 1e-3
  maxdims = [3, 5, 8, 10, 11, 12, 13, 14, 15, 16, 20, 31, 32]
  for dim in maxdims
    size = dim * dim * linkdim
    out, out2 = benchmark_3D_contraction(tn; cutoff=cutoff, maxdim=dim, maxsize=size)
    error1 = abs((out - out1) / out1)
    error2 = abs((out2 - out1) / out1)
    print("maxdim, ", dim, ", error1, ", error1, ", error2, ", error2, "\n")
  end
end

@testset "test 3-D cube with 1D grouping" begin
  ITensors.set_warn_order(100)
  do_profile(true)
  N = (3, 3, 3) # (5, 5, 5)
  linkdim = 2
  maxdim = linkdim^(floor(N[1] * N[2]))
  cutoff = 1e-15
  # tn = ising_partition(N, linkdim)
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)
  tn = map(inds -> randomITensor(inds...), tn_inds)
  tn = reshape(tn, (N[1], N[2] * N[3]))
  tntree = tn[:, 1]
  for i in 2:(N[2] * N[3])
    tntree = [tntree, tn[:, i]]
  end
  approximate_contract(
    tntree; cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mincut"
  )
  profile_exit()
end

#TODO
# @testset "test 3-D cube with DMRG-like algorithm" begin
#   ITensors.set_warn_order(100)
#   do_profile(true)
#   N = (5, 5, 3) # (5, 5, 5)
#   linkdim = 2
#   maxdim = linkdim^(floor(N[1] * N[2]))
#   cutoff = 1e-15
#   tn = ising_partition(N, linkdim)
#   function build_tree(i)
#     tree = tn[:, 1, i]
#     for j in 2:N[2]
#       tree = [tree, tn[:, j, i]]
#     end
#     return tree
#   end
#   tn1, _ = approximate_contract(
#     build_tree(1); cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
#   )
#   tn2, _ = approximate_contract(
#     build_tree(2); cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
#   )
#   tn3, _ = approximate_contract(
#     build_tree(3); cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
#   )
#   tn12, _ = approximate_contract(
#     [tn1..., tn2...]; cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
#   )
#   out, _ = approximate_contract(
#     [tn12..., tn3...]; cutoff=cutoff, maxdim=maxdim, maxsize=1e15, algorithm="mps"
#   )
# profile_exit()
# end

@testset "benchmark PEPS" begin
  N = (8, 8) #(12, 12)
  linkdim = 10
  cutoff = 1e-15
  tn_inds = inds_network(N...; linkdims=linkdim, periodic=false)

  dim = 20
  size = dim * dim * linkdim
  # warmup
  for i in 1:2
    tn = map(inds -> randomITensor(inds...), tn_inds)
    ITensors.set_warn_order(100)
    benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim, maxsize=size)
  end

  do_profile(true)
  for i in 1:3
    tn = map(inds -> randomITensor(inds...), tn_inds)
    ITensors.set_warn_order(100)
    benchmark_peps_contraction(tn; cutoff=cutoff, maxdim=dim, maxsize=size)
  end
  profile_exit()
end
