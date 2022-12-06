using ITensors, Random, SweepContractor
using ITensorNetworks.ApproximateTNContraction:
  timer, ITensor_networks, line_network, approximate_contract

include("utils.jl")

@testset "test the interface" begin
  LTN = LabelledTensorNetwork{Char}()
  LTN['A'] = Tensor(['D', 'B'], [i^2 - 2j for i in 0:2, j in 0:2], 0, 1)
  LTN['B'] = Tensor(['A', 'D', 'C'], [-3^i * j + k for i in 0:2, j in 0:2, k in 0:2], 0, 0)
  LTN['C'] = Tensor(['B', 'D'], [j for i in 0:2, j in 0:2], 1, 0)
  LTN['D'] = Tensor(['A', 'B', 'C'], [i * j * k for i in 0:2, j in 0:2, k in 0:2], 1, 1)

  sweep = sweep_contract(LTN, 100, 100; fast=true)
  out = ldexp(sweep...)
  @test isapprox(out, contract(ITensor_networks(LTN))[])
  show(timer)
end

function lattice(row, column, d)
  function build_adj(i, j)
    adj = Vector{Int64}()
    i > 1 && push!(adj, delabel[(i - 1, j)])
    i < row && push!(adj, delabel[(i + 1, j)])
    j > 1 && push!(adj, delabel[(i, j - 1)])
    j < column && push!(adj, delabel[(i, j + 1)])
    return adj
  end
  TN = TensorNetwork()
  delabel = Dict()
  index = 1
  for i in 1:row
    ranges = iseven(i) ? (column:-1:1) : (1:column)
    for j in ranges
      delabel[(i, j)] = index
      index += 1
    end
  end
  for i in 1:row
    ranges = iseven(i) ? (column:-1:1) : (1:column)
    for j in ranges
      adj = build_adj(i, j)
      push!(TN, Tensor(adj, randn(d * ones(Int, length(adj))...), i, j))
    end
  end
  return TN
end

function get_contracted_peps(LTN, rank, N)
  tnet = ITensor_networks(LTN)
  tnet_mat = reshape(tnet, N...)
  out_mps = peps_contraction_mpomps(tnet_mat; cutoff=1e-15, maxdim=rank, snake=true)
  out = contract_w_sweep(LTN, rank)
  out2 = contract_element_group(tnet; cutoff=1e-15, maxdim=rank)
  out3 = contract_line_group(tnet_mat; cutoff=1e-15, maxdim=rank)
  return out, out2, out3, out_mps[]
end

@testset "test on 2D grid" begin
  Random.seed!(1234)
  ITensors.set_warn_order(100)
  row, column, d = 8, 8, 2
  LTN = lattice(row, column, d)

  out_true, out_element, out_line, out_mps = get_contracted_peps(
    LTN, d^(Int(row / 2)), [row, column]
  )
  @test abs((out_true - out_element) / out_true) < 1e-3
  @test abs((out_true - out_line) / out_true) < 1e-3
  @test abs((out_true - out_mps) / out_true) < 1e-3
  for rank in [2, 3, 4, 6, 8, 10, 12, 14, 15, 16]
    out, out_element, out_line, out_mps = get_contracted_peps(LTN, rank, [row, column])
    error_sweepcontractor = abs((out - out_true) / out_true)
    error_element = abs((out_element - out_true) / out_true)
    error_line = abs((out_line - out_true) / out_true)
    error_mps = abs((out_mps - out_true) / out_true)
    print(
      "maxdim, ",
      rank,
      ", error_sweepcontractor, ",
      error_sweepcontractor,
      ", error_element, ",
      error_element,
      ", error_line, ",
      error_line,
      ", error_mps, ",
      error_mps,
      "\n",
    )
  end
end

@testset "benchmark on 2D grid" begin
  Random.seed!(1234)
  ITensors.set_warn_order(100)
  row, column, d, rank = 10, 10, 2, 10
  LTN = lattice(row, column, d)
  # warm-up
  get_contracted_peps(LTN, rank, [row, column])
  @info "start benchmark on 2D grid"
  reset_timer!(timer)
  LTN = lattice(row, column, d)
  get_contracted_peps(LTN, rank, [row, column])
  show(timer)
end

function cube_3d(L=3, d=2)
  function build_adj(i, j, k)
    adj = Vector{Int64}()
    i > 1 && push!(adj, delabel[(i - 1, j, k)])
    i < L && push!(adj, delabel[(i + 1, j, k)])
    j > 1 && push!(adj, delabel[(i, j - 1, k)])
    j < L && push!(adj, delabel[(i, j + 1, k)])
    k > 1 && push!(adj, delabel[(i, j, k - 1)])
    k < L && push!(adj, delabel[(i, j, k + 1)])
    return adj
  end
  TN = TensorNetwork()
  delabel = Dict()
  index = 1
  for i in 1:L
    ranges_j = iseven(i) ? (L:-1:1) : (1:L)
    for j in ranges_j
      ranges_k = iseven((i - 1) * L + j) ? (L:-1:1) : (1:L)
      for k in ranges_k
        delabel[(i, j, k)] = index
        index += 1
      end
    end
  end
  for i in 1:L
    ranges_j = iseven(i) ? (L:-1:1) : (1:L)
    for j in ranges_j
      ranges_k = iseven((i - 1) * L + j) ? (L:-1:1) : (1:L)
      for k in ranges_k
        adj = build_adj(i, j, k)
        newt = Tensor(
          adj, randn(d * ones(Int, length(adj))...), i + 0.01 * randn(), j + 0.01 * randn()
        )
        push!(TN, newt)
      end
    end
  end
  return TN
end

@testset "test on 3D cube with element grouping" begin
  Random.seed!(1234)
  ITensors.set_warn_order(100)
  L, d = 3, 2
  rank = 16
  TN = cube_3d(L, d)
  out = contract_w_sweep(TN, rank)
  tnet = ITensor_networks(TN)
  out2 = contract_element_group(tnet; cutoff=1e-15, maxdim=rank)

  reset_timer!(timer)
  TN = cube_3d(L, d)
  out = contract_w_sweep(TN, rank)
  tnet = ITensor_networks(TN)
  out2 = contract_element_group(tnet; cutoff=1e-15, maxdim=rank)
  show(timer)
end
