module ModelHamiltonians
using Dictionaries: Dictionary
using Graphs: AbstractGraph, dst, edges, edgetype, neighborhood, path_graph, src, vertices
using ITensors.Ops: OpSum

to_callable(value::Type) = value
to_callable(value::Function) = value
to_callable(value::Dict) = Base.Fix1(getindex, value)
to_callable(value::Dictionary) = Base.Fix1(getindex, value)
to_callable(value) = Returns(value)

# TODO: Move to `NamedGraphs.jl` or `GraphsExtensions.jl`.
# TODO: Add a tet for this.
function nth_nearest_neighbors(g, v, n::Int)
  isone(n) && return neighborhood(g, v, 1)
  return setdiff(neighborhood(g, v, n), neighborhood(g, v, n - 1))
end

# TODO: Move to `NamedGraphs.jl` or `GraphsExtensions.jl`.
# TODO: Add a tet for this.
next_nearest_neighbors(g, v) = nth_nearest_neighbors(g, v, 2)

function tight_binding(g::AbstractGraph; t=1, tp=0, h=0)
  (; t, tp, h) = map(to_callable, (; t, tp, h))
  h = to_callable(h)
  ℋ = OpSum()
  for e in edges(g)
    ℋ -= t(e), "Cdag", src(e), "C", dst(e)
    ℋ -= t(e), "Cdag", dst(e), "C", src(e)
  end
  for v in vertices(g)
    for nn in next_nearest_neighbors(g, v)
      e = edgetype(g)(v, nn)
      ℋ -= tp(e), "Cdag", v, "C", nn
      ℋ -= tp(e), "Cdag", nn, "C", v
    end
  end
  for v in vertices(g)
    ℋ -= h(v), "N", v
  end
  return ℋ
end

"""
t-t' Hubbard Model g,i,v
"""
function hubbard(g::AbstractGraph; U=0, t=1, tp=0, h=0)
  (; U, t, tp, h) = map(to_callable, (; U, t, tp, h))
  ℋ = OpSum()
  for e in edges(g)
    ℋ -= t(e), "Cdagup", src(e), "Cup", dst(e)
    ℋ -= t(e), "Cdagup", dst(e), "Cup", src(e)
    ℋ -= t(e), "Cdagdn", src(e), "Cdn", dst(e)
    ℋ -= t(e), "Cdagdn", dst(e), "Cdn", src(e)
  end
  for v in vertices(g)
    for nn in next_nearest_neighbors(g, v)
      e = edgetype(g)(v, nn)
      ℋ -= tp(e), "Cdagup", v, "Cup", nn
      ℋ -= tp(e), "Cdagup", nn, "Cup", v
      ℋ -= tp(e), "Cdagdn", v, "Cdn", nn
      ℋ -= tp(e), "Cdagdn", nn, "Cdn", v
    end
  end
  ℋ -= h(v), "Sz", v
  ℋ += U(v), "Nupdn", v
  return ℋ
end

"""
Random field J1-J2 Heisenberg model on a general graph
"""
function heisenberg(g::AbstractGraph; J1=1, J2=0, h=0)
  (; J1, J2, h) = map(to_callable, (; J1, J2, h))
  ℋ = OpSum()
  for e in edges(g)
    ℋ += J1(e) / 2, "S+", src(e), "S-", dst(e)
    ℋ += J1(e) / 2, "S-", src(e), "S+", dst(e)
    ℋ += J1(e), "Sz", src(e), "Sz", dst(e)
  end
  for v in vertices(g)
    for nn in next_nearest_neighbors(g, v)
      e = edgetype(g)(v, nn)
      ℋ += J2(e) / 2, "S+", v, "S-", nn
      ℋ += J2(e) / 2, "S-", v, "S+", nn
      ℋ += J2(e), "Sz", v, "Sz", nn
    end
  end
  for v in vertices(g)
    ℋ += h(v), "Sz", v
  end
  return ℋ
end

"""
Next-to-nearest-neighbor Ising model (ZZX) on a general graph
"""
function ising(g::AbstractGraph; J1=-1, J2=0, h=0)
  # TODO: Make `J2` callable once other issues
  # are addressed.
  (; J1, h) = map(to_callable, (; J1, h))
  ℋ = OpSum()
  for e in edges(g)
    ℋ += J1(e), "Sz", src(e), "Sz", dst(e)
  end
  for v in vertices(g)
    # TODO: Try removing this if-statement. This
    # helps to avoid constructing next-nearest
    # neighbor gates, and also avoids a bug
    # in `neighborhood` for integer labels.
    if !iszero(J2)
      for nn in next_nearest_neighbors(g, v)
        e = edgetype(g)(v, nn)
        ℋ += J2, "Sz", v, "Sz", nn
      end
    end
  end
  for v in vertices(g)
    ℋ += h(v), "Sx", v
  end
  return ℋ
end

"""
Random field J1-J2 Heisenberg model on a chain of length N
"""
heisenberg(N::Integer; kwargs...) = heisenberg(path_graph(N); kwargs...)

"""
Next-to-nearest-neighbor Ising model (ZZX) on a chain of length N
"""
ising(N::Integer; kwargs...) = ising(path_graph(N); kwargs...)
end
