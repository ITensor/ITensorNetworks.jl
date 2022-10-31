using Graphs: AbstractGraph, grid
using Dictionaries: AbstractDictionary

_maybe_fill(x, n) = x
_maybe_fill(x::Number, n) = fill(x, n)

"""
Random field J1-J2 Heisenberg model on a general graph
"""
function heisenberg_graph(
  g::AbstractGraph; J1=1.0, J2=0.0, h::Union{<:Real,Vector{<:Real}}=0
)
  h = _maybe_fill(h, nv(g))
  H = OpSum()
  if !iszero(J1)
    for e in edges(g)
      H += J1 / 2, "S+", src(e), "S-", dst(e)
      H += J1 / 2, "S-", src(e), "S+", dst(e)
      H += J1, "Sz", src(e), "Sz", dst(e)
    end
  end
  if !iszero(J2)
    # TODO, more clever way of looping over next to nearest neighbors?
    for (i, v) in enumerate(vertices(g))
      nnn = [neighbors(g, n) for n in neighbors(g, v)]
      nnn = setdiff(Base.Iterators.flatten(nnn), neighbors(g, v))
      nnn = setdiff(nnn, vertices(g)[1:i])
      for nn in nnn
        H += J2 / 2, "S+", v, "S-", nn
        H += J2 / 2, "S-", v, "S+", nn
        H += J2, "Sz", v, "Sz", nn
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      H -= h[i], "Sz", v
    end
  end
  return H
end

"""
Next-to-nearest-neighbor Ising model (ZZX) on a general graph
"""
function ising_graph(g::AbstractGraph; J1=-1.0, J2=0.0, h::Union{<:Real,Vector{<:Real}}=0)
  h = _maybe_fill(h, nv(g))
  H = OpSum()
  if !iszero(J1)
    for e in edges(g)
      H += J1, "Z", src(e), "Z", dst(e)
    end
  end
  if !iszero(J2)
    # TODO, more clever way of looping over next to nearest neighbors?
    for (i, v) in enumerate(vertices(g))
      nnn = [neighbors(g, n) for n in neighbors(g, v)]
      nnn = setdiff(Base.Iterators.flatten(nnn), neighbors(g, v))
      nnn = setdiff(nnn, vertices(g)[1:i])
      for nn in nnn
        H += J2, "Z", v, "Z", nn
      end
    end
  end
  for (i, v) in enumerate(vertices(g))
    if !iszero(h[i])
      H += h[i], "X", v
    end
  end
  return H
end

"""
Random field J1-J2 Heisenberg model on a chain of length N
"""
heisenberg(N; kwargs...) = heisenberg_graph(grid((N,)); kwargs...)

"""
Next-to-nearest-neighbor Ising model (ZZX) on a chain of length N
"""
ising(N; kwargs...) = ising(grid((N,)); kwargs...)
