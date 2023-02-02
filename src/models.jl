_maybe_fill(x, n) = x
_maybe_fill(x::Number, n) = fill(x, n)

"""
Random field J1-J2 Heisenberg model on a general graph
"""
function heisenberg(g::AbstractGraph; J1=1.0, J2=0.0, h::Union{<:Real,Vector{<:Real}}=0)
  h = _maybe_fill(h, nv(g))
  ℋ = OpSum()
  if !iszero(J1)
    for e in edges(g)
      ℋ += J1 / 2, "S+", maybe_only(src(e)), "S-", maybe_only(dst(e))
      ℋ += J1 / 2, "S-", maybe_only(src(e)), "S+", maybe_only(dst(e))
      ℋ += J1, "Sz", maybe_only(src(e)), "Sz", maybe_only(dst(e))
    end
  end
  if !iszero(J2)
    # TODO, more clever way of looping over next to nearest neighbors?
    for (i, v) in enumerate(vertices(g))
      nnn = [neighbors(g, n) for n in neighbors(g, v)]
      nnn = setdiff(Base.Iterators.flatten(nnn), neighbors(g, v))
      nnn = setdiff(nnn, vertices(g)[1:i])
      for nn in nnn
        ℋ += J2 / 2, "S+", maybe_only(v), "S-", maybe_only(nn)
        ℋ += J2 / 2, "S-", maybe_only(v), "S+", maybe_only(nn)
        ℋ += J2, "Sz", maybe_only(v), "Sz", maybe_only(nn)
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ -= h[i], "Sz", maybe_only(v)
    end
  end
  return ℋ
end

"""
Next-to-nearest-neighbor Ising model (ZZX) on a general graph
"""
function ising(g::AbstractGraph; J1=-1.0, J2=0.0, h::Union{<:Real,Vector{<:Real}}=0)
  h = _maybe_fill(h, nv(g))
  ℋ = OpSum()
  if !iszero(J1)
    for e in edges(g)
      ℋ += J1, "Z", maybe_only(src(e)), "Z", maybe_only(dst(e))
    end
  end
  if !iszero(J2)
    # TODO, more clever way of looping over next to nearest neighbors?
    for (i, v) in enumerate(vertices(g))
      nnn = [neighbors(g, n) for n in neighbors(g, v)]
      nnn = setdiff(Base.Iterators.flatten(nnn), neighbors(g, v))
      nnn = setdiff(nnn, vertices(g)[1:i])
      for nn in nnn
        ℋ += J2, "Z", maybe_only(v), "Z", maybe_only(nn)
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ += h[i], "X", maybe_only(v)
    end
  end
  return ℋ
end

"""
Random field J1-J2 Heisenberg model on a chain of length N
"""
heisenberg(N::Integer; kwargs...) = heisenberg(grid((N,)); kwargs...)

"""
Next-to-nearest-neighbor Ising model (ZZX) on a chain of length N
"""
ising(N::Integer; kwargs...) = ising(grid((N,)); kwargs...)
