_maybe_fill(x, n) = x
_maybe_fill(x::Number, n) = fill(x, n)

#= Make the free fermion Hamiltonian for the up spins
os_up = OpSum()
for n in 1:(N - 1)
  os_up .+= -t, "Cdagup", n, "Cup", n + 1
  os_up .+= -t, "Cdagup", n + 1, "Cup", n
end

# Make the free fermion Hamiltonian for the down spins
os_dn = OpSum()
for n in 1:(N - 1)
  os_dn .+= -t, "Cdagdn", n, "Cdn", n + 1
  os_dn .+= -t, "Cdagdn", n + 1, "Cdn", n
end
=#
function tight_binding(g::AbstractGraph; t=1.0, tp=0.0, h::Union{<:Real,Vector{<:Real}}=0)
  h = _maybe_fill(h, nv(g))
  ℋ = OpSum()
  if !iszero(t)
    for e in edges(g)
      ℋ += -t, "Cdag", maybe_only(src(e)), "C", maybe_only(dst(e))
      ℋ += -t, "Cdag", maybe_only(dst(e)), "C", maybe_only(src(e))
    end
  end
  if !iszero(t')
    # TODO, more clever way of looping over next to nearest neighbors?
    for (i, v) in enumerate(vertices(g))
      nnn = [neighbors(g, n) for n in neighbors(g, v)]
      nnn = setdiff(Base.Iterators.flatten(nnn), neighbors(g, v))
      nnn = setdiff(nnn, vertices(g)[1:i])
      for nn in nnn
        ℋ += -tp, "Cdag", maybe_only(v), "C", maybe_only(nn)
        ℋ += -tp, "Cdag", maybe_only(nn), "C", maybe_only(v)
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ -= h[i], "N", maybe_only(v)
    end
  end
  return ℋ
end

"""
t-t' Hubbard Model 
"""
function hubbard(g::AbstractGraph; U=0.0, t=1.0, tp=0.0, h::Union{<:Real,Vector{<:Real}}=0)
  h = _maybe_fill(h, nv(g))
  ℋ = OpSum()
  if !iszero(t)
    for e in edges(g)
      ℋ += -t, "Cdagup", maybe_only(src(e)), "Cup", maybe_only(dst(e))
      ℋ += -t, "Cdagup", maybe_only(dst(e)), "Cup", maybe_only(src(e))
      ℋ += -t, "Cdagdn", maybe_only(src(e)), "Cdn", maybe_only(dst(e))
      ℋ += -t, "Cdagdn", maybe_only(dst(e)), "Cdn", maybe_only(src(e))
    end
  end
  if !iszero(tp)
    # TODO, more clever way of looping over next to nearest neighbors?
    for (i, v) in enumerate(vertices(g))
      nnn = [neighbors(g, n) for n in neighbors(g, v)]
      nnn = setdiff(Base.Iterators.flatten(nnn), neighbors(g, v))
      nnn = setdiff(nnn, vertices(g)[1:i])
      for nn in nnn
        ℋ += -tp, "Cdagup", maybe_only(v), "Cup", maybe_only(nn)
        ℋ += -tp, "Cdagup", maybe_only(nn), "Cup", maybe_only(v)
        ℋ += -tp, "Cdagdn", maybe_only(v), "Cdn", maybe_only(nn)
        ℋ += -tp, "Cdagdn", maybe_only(nn), "Cdn", maybe_only(v)
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ -= h[i], "Sz", maybe_only(v)
    end
    if !iszero(U)
      ℋ += U, "Nupdn", maybe_only(v)
    end
  end
  return ℋ
end

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
        ℋ += J2, "Sz", maybe_only(v), "Sz", maybe_only(nn)
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ += h[i], "Sx", maybe_only(v)
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
