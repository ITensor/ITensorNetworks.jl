using ITensorNetworkAD
using AutoHOOT, ITensors, Zygote
using ITensorNetworkAD.ITensorNetworks: GeneralMPSTensor

const itensorah = ITensorNetworkAD.ITensorAutoHOOT

@testset "test MPSTensor" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  l = Index(4, "l")
  m = Index(5, "m")

  A = randomITensor(i, j, k)
  B = randomITensor(k, l, m)
  C = randomITensor(i, j, l, m)
  mps_A = GeneralMPSTensor(MPS(A, inds(A)))
  mps_B = GeneralMPSTensor(MPS(B, inds(B)))
  mps_C = GeneralMPSTensor(MPS(C, inds(C)))

  out = A * B
  network = [mps_A, mps_B]
  nodes, dict = itensorah.generate_einsum_expr([network])
  out_list = itensorah.compute_graph(nodes, dict; cutoff=1e-15, maxdim=1000)
  @test isapprox(out, ITensor(out_list[1]))

  out = A * B * C
  out2 = contract(mps_A, mps_B, mps_C; cutoff=1e-15, maxdim=1000)
  @test isapprox(out, ITensor(out2))
end

@testset "test batch_tensor_contraction" begin
  i = Index(2, "i")
  j = Index(3, "j")
  k = Index(2, "k")
  A = randomITensor(i, j)
  B = randomITensor(j, k)
  C = randomITensor(k, i)

  function network(A)
    mps_A = GeneralMPSTensor(A; cutoff=1e-15, maxdim=1000)
    mps_B = GeneralMPSTensor(B; cutoff=1e-15, maxdim=1000)
    mps_C = GeneralMPSTensor(C; cutoff=1e-15, maxdim=1000)
    tensor_network = [mps_A, mps_B, mps_C]
    out = itensorah.batch_tensor_contraction(
      [tensor_network], mps_A; cutoff=1e-15, maxdim=1000
    )
    return sum(out)[]
  end
  grad_A = gradient(network, A)
  @test isapprox(grad_A[1], B * C)
end
