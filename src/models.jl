function ising(g::AbstractGraph; h)
  ℋ = OpSum()
  for e in edges(g)
    ℋ -= "Z", maybe_only(src(e)), "Z", maybe_only(dst(e))
  end
  for v in vertices(g)
    ℋ += h, "X", maybe_only(v)
  end
  return ℋ
end
