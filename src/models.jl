using Graphs: grid, neighborhood, vertices
using ITensors.Ops: OpSum

_maybe_fill(x, n) = x
_maybe_fill(x::Number, n) = fill(x, n)

function nth_nearest_neighbors(g, v, n::Int)  #ToDo: Add test for this.
  isone(n) && return neighborhood(g, v, 1)
  return setdiff(neighborhood(g, v, n), neighborhood(g, v, n - 1))
end

# TODO: Move to `NamedGraphs.jl` or `GraphsExtensions.jl`.
next_nearest_neighbors(g, v) = nth_nearest_neighbors(g, v, 2)

function tight_binding(g::AbstractGraph; t=1, tp=0, h=0)
  h = _maybe_fill(h, nv(g))
  ℋ = OpSum()
  if !iszero(t)
    for e in edges(g)
      ℋ -= t, "Cdag", src(e), "C", dst(e)
      ℋ -= t, "Cdag", dst(e), "C", src(e)
    end
  end
  if !iszero(t')
    for (i, v) in enumerate(vertices(g))
      for nn in next_nearest_neighbors(g, v)
        ℋ -= tp, "Cdag", v, "C", nn
        ℋ -= tp, "Cdag", nn, "C", v
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ -= h[i], "N", v
    end
  end
  return ℋ
end

"""
t-t' Hubbard Model g,i,v
"""
function hubbard(g::AbstractGraph; U=0, t=1, tp=0, h=0)
  h = _maybe_fill(h, nv(g))
  ℋ = OpSum()
  if !iszero(t)
    for e in edges(g)
      ℋ -= t, "Cdagup", src(e), "Cup", dst(e)
      ℋ -= t, "Cdagup", dst(e), "Cup", src(e)
      ℋ -= t, "Cdagdn", src(e), "Cdn", dst(e)
      ℋ -= t, "Cdagdn", dst(e), "Cdn", src(e)
    end
  end
  if !iszero(tp)
    # TODO, more clever way of looping over next to nearest neighbors?
    for (i, v) in enumerate(vertices(g))
      for nn in next_nearest_neighbors(g, v)
        ℋ -= tp, "Cdagup", v, "Cup", nn
        ℋ -= tp, "Cdagup", nn, "Cup", v
        ℋ -= tp, "Cdagdn", v, "Cdn", nn
        ℋ -= tp, "Cdagdn", nn, "Cdn", v
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ -= h[i], "Sz", v
    end
    if !iszero(U)
      ℋ += U, "Nupdn", v
    end
  end
  return ℋ
end

"""
Random field J1-J2 Heisenberg model on a general graph
"""
function heisenberg(g::AbstractGraph; J1=1, J2=0, h=0)
  h = _maybe_fill(h, nv(g))
  ℋ = OpSum()
  if !iszero(J1)
    for e in edges(g)
      ℋ += J1 / 2, "S+", src(e), "S-", dst(e)
      ℋ += J1 / 2, "S-", src(e), "S+", dst(e)
      ℋ += J1, "Sz", src(e), "Sz", dst(e)
    end
  end
  if !iszero(J2)
    for (i, v) in enumerate(vertices(g))
      for nn in next_nearest_neighbors(g, v)
        ℋ += J2 / 2, "S+", v, "S-", nn
        ℋ += J2 / 2, "S-", v, "S+", nn
        ℋ += J2, "Sz", v, "Sz", nn
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ += h[i], "Sz", v
    end
  end
  return ℋ
end

"""
Next-to-nearest-neighbor Ising model (ZZX) on a general graph
"""
function ising(g::AbstractGraph; J1=-1, J2=0, h=0)
  h = _maybe_fill(h, nv(g))
  ℋ = OpSum()
  if !iszero(J1)
    for e in edges(g)
      ℋ += J1, "Sz", src(e), "Sz", dst(e)
    end
  end
  if !iszero(J2)
    for (i, v) in enumerate(vertices(g))
      for nn in next_nearest_neighbors(g, v)
        ℋ += J2, "Sz", v, "Sz", nn
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      ℋ += h[i], "Sx", v
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
