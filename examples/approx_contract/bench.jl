using ITensorNetworks

function bench_lnZ(
  tntree;
  num_iter,
  cutoff,
  maxdim,
  ansatz,
  approx_itensornetwork_alg,
  swap_size,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
  linear_ordering_alg,
)
  reset_timer!(ITensors.timer)
  function _run()
    out, log_acc_norm = partitioned_contract(
      tntree;
      cutoff,
      maxdim,
      ansatz,
      approx_itensornetwork_alg,
      swap_size,
      contraction_sequence_alg,
      contraction_sequence_kwargs,
      linear_ordering_alg,
    )
    out = Vector{ITensor}(out)
    log_acc_norm = log(norm(out)) + log_acc_norm
    @info "out value is", out[1].tensor
    @info "out norm is", log_acc_norm
    return log_acc_norm
  end
  out_list = []
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  show(ITensors.timer)
  # after warmup, start to benchmark
  reset_timer!(ITensors.timer)
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  @info "lnZ results are", out_list, "mean is", sum(out_list) / (num_iter * 2)
  return show(ITensors.timer)
end

function bench_magnetization(
  tntree_pair;
  num_iter,
  cutoff,
  maxdim,
  ansatz,
  approx_itensornetwork_alg,
  swap_size,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
  linear_ordering_alg,
  warmup,
)
  reset_timer!(ITensors.timer)
  tntree1, tntree2 = tntree_pair
  function _run()
    out, log_acc_norm = approximate_contract(
      tntree1;
      cutoff,
      maxdim,
      ansatz,
      approx_itensornetwork_alg,
      swap_size,
      contraction_sequence_alg,
      contraction_sequence_kwargs,
      linear_ordering_alg,
    )
    out = Vector{ITensor}(out)
    lognorm1 = log(norm(out)) + log_acc_norm
    out, log_acc_norm = approximate_contract(
      tntree2;
      cutoff,
      maxdim,
      ansatz,
      approx_itensornetwork_alg,
      swap_size,
      contraction_sequence_alg,
      contraction_sequence_kwargs,
      linear_ordering_alg,
    )
    out = Vector{ITensor}(out)
    lognorm2 = log(norm(out)) + log_acc_norm
    @info "magnetization is", exp(lognorm1 - lognorm2)
    return exp(lognorm1 - lognorm2)
  end
  out_list = []
  total_iter = num_iter
  if warmup
    for _ in 1:num_iter
      push!(out_list, _run())
    end
    show(ITensors.timer)
    total_iter += num_iter
  end
  @info "warmup finished"
  # after warmup, start to benchmark
  reset_timer!(ITensors.timer)
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  @info "magnetization results are", out_list, "mean is", sum(out_list) / total_iter
  return show(ITensors.timer)
end
