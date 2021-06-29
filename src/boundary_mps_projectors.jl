
function truncation_projectors(H::Vector{ITensor}, ψ::Vector{ITensor}; kwargs...)
  return truncation_projectors(MPO(H), MPS(ψ); kwargs...)
end

maxdim_arg(T::Type) = round(Int, Sys.total_memory() / sizeof(T), RoundDown)
_maxdim_arg(a) = maxdim_arg(mapreduce(eltype, promote_type, a))
maxdim_arg(a::AbstractArray) = _maxdim_arg(a)
maxdim_arg(a::Union{MPS,MPO}) = _maxdim_arg(a)

function truncation_projectors(H::MPO, ψ::MPS; split_tags=("" => ""), split_plevs=(0 => 1), cutoff=1e-8, maxdim=maxdim_arg(H), tol=1e-6, maxiter=10)
  left_tags, right_tags = split_tags
  left_plev, right_plev = split_plevs
  N = length(ψ)
  @assert length(H) == N
  ψd = dag(ψ)
  Hd = dag(H)
  ψd = sim(linkinds, ψd)
  Hd = sim(linkinds, Hd)
  Hd, ψd = sim(siteinds, commoninds, Hd, ψd)
  linkindsH = linkinds(H)
  new_tags = tags.(linkindsH)
  new_plevs = plev.(linkindsH)
  L = Vector{ITensor}(undef, N)
  L[1] = ψ[1] * H[1] * Hd[1] * ψd[1]
  normalize!(L[1])
  for n in 2:N
    L[n] = L[n - 1] * ψ[n] * H[n] * Hd[n] * ψd[n]
    normalize!(L[n])
  end
  R = Vector{ITensor}(undef, N)
  R[N - 1] = ψ[N] * H[N] * Hd[N] * ψd[N]
  normalize!(R[N - 1])
  for n in reverse(2:(N - 1))
    R[n - 1] = R[n] * ψ[n] * H[n] * Hd[n] * ψd[n]
    normalize!(R[n - 1])
  end
  Us = Vector{ITensor}(undef, N - 1)
  Uds = Vector{ITensor}(undef, N - 1)
  for n in 1:(N - 1)
    link_ψn = commonind(ψ[n], ψ[n + 1])
    link_Hn = commonind(H[n], H[n + 1])
    links_n = (link_ψn, link_Hn)
    left_links_n = setprime(addtags(links_n, left_tags), left_plev)
    right_links_n = setprime(addtags(links_n, right_tags), right_plev)
    Ln = replaceinds(L[n], links_n => left_links_n)
    Rn = replaceinds(R[n], links_n => right_links_n)
    ρn = Ln * Rn
    U, S, V, spec = svd(ρn, left_links_n; maxdim=maxdim, cutoff=cutoff, lefttags=new_tags[n], leftplev=new_plevs[n])
    Ud = replaceinds(dag(U), left_links_n => right_links_n)
    overlap_old = (ρn * dag(U) * dag(Ud))[]
    err = 1.0
    iter = 0
    while err > tol && iter ≤ maxiter
      ΛU = dag(ρn * dag(Ud))
      U, P = polar(ΛU, left_links_n)
      u = commonind(U, P)
      setprime!(U, new_plevs[n]; inds=u)
      Ud = replaceinds(dag(U), left_links_n => right_links_n)
      overlap_new = (ρn * U * Ud)[]
      err = abs(overlap_new - overlap_old) / overlap_new
      iter += 1
    end
    if err > tol
      @warn "In $iter iterations, computing projector between sites $n and $(n + 1) gave a truncation error of $err, which is larger than the tolerance $tol."
    end
    Us[n] = U
    Uds[n] = Ud
  end
  return Us .=> Uds
end

# Assume prime level convention for site indices of MPO of dag(s) -> s'
function truncation_projectors_assume_primes(H::MPO, ψ::MPS; cutoff=1e-8)
  N = length(ψ)
  @assert length(H) == N
  ψd = dag(ψ)''
  # TODO: add swapprime(siteinds, H, 0 => 1)
  Hd = swapprime(dag(H), 0 => 1)'
  L = Vector{ITensor}(undef, N)
  L[1] = ψ[1] * H[1] * Hd[1] * ψd[1]
  for n in 2:N
    L[n] = L[n - 1] * ψ[n] * H[n] * Hd[n] * ψd[n]
  end
  R = Vector{ITensor}(undef, N)
  R[N - 1] = ψ[N] * H[N] * Hd[N] * ψd[N]
  for n in reverse(2:(N - 1))
    R[n - 1] = R[n] * ψ[n] * H[n] * Hd[n] * ψd[n]
  end

  for n in 1:(N - 1)
    left_tags = "left"
    right_tags = "right"
    link_ψn = commonind(ψ[n], ψ[n + 1])
    link_Hn = commonind(H[n], H[n + 1])
    links_n = (link_ψn, link_Hn)
    left_links_n = addtags(links_n, left_tags)
    right_links_n = addtags(links_n, right_tags)
    Ln = addtags(L[n], left_tags; inds=links_n)
    Rn = addtags(R[n], right_tags; inds=links_n)
    ρn = Ln * Rn
    U, S, V, spec = svd(ρn, left_links_n; cutoff=cutoff)
    Ud = replacetags(dag(U), left_tags => right_tags)
    for _ in 1:10
      ΛU = dag(ρn * Ud)
      U, _ = polar(ΛU, left_links_n)
      Ud = replacetags(dag(U), left_tags => right_tags)
    end
  end
end

