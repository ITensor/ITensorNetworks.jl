using ITensors.LazyApply: Applied, Prod, Scaled, Sum
using ITensors.Ops: Ops, Op

# TODO: Rename this `replace_sites`?
# TODO: Use `fmap`, `deepmap`, `treemap`?
function replace_vertices(f, ∑o::Sum)
  return Sum(map(oᵢ -> replace_vertices(f, oᵢ), Ops.terms(∑o)))
end

function replace_vertices(f, ∏o::Prod)
  return Prod(map(oᵢ -> replace_vertices(f, oᵢ), Ops.terms(∏o)))
end

function replace_vertices(f, o::Scaled)
  return maybe_real(Ops.coefficient(o)) * replace_vertices(f, Ops.argument(o))
end

set_sites(o::Op, sites) = Op(Ops.which_op(o), sites...; Ops.params(o)...)

function replace_vertices(f, o::Op)
  return set_sites(o, f.(Ops.sites(o)))
end

function replace_vertices(o::Union{Op,Applied}, vertex_map)
  return replace_vertices(v -> get(vertex_map, v, v), o)
end

function group_terms(ℋ::Sum, g)
  grouped_terms = group(ITensors.terms(ℋ)) do t
    findfirst(edges(g)) do e
      to_tuple.(ITensors.sites(t)) ⊆ [src(e), dst(e)]
    end
  end
  return Sum(collect(sum.(grouped_terms)))
end

function ITensors.ITensor(o::Op, s::IndsNetwork)
  s⃗ = [only(s[nᵢ]) for nᵢ in Ops.sites(o)]
  return op(Ops.which_op(o), s⃗; Ops.params(o)...)
end

function ITensors.ITensor(∏o::Prod, s::IndsNetwork)
  T = ITensor(1.0)
  for oᵢ in Ops.terms(∏o)
    Tᵢ = ITensor(oᵢ, s)
    # For now, only support operators on distinct
    # sites.
    @assert !hascommoninds(T, Tᵢ)
    T *= Tᵢ
  end
  return T
end

function ITensors.ITensor(∑o::Sum, s::IndsNetwork)
  T = ITensor(0)
  for oᵢ in Ops.terms(∑o)
    Tᵢ = ITensor(oᵢ, s)
    T = tensor_sum(T, Tᵢ)
  end
  return T
end

function ITensors.ITensor(o::Scaled, s::IndsNetwork)
  return maybe_real(Ops.coefficient(o)) * ITensor(Ops.argument(o), s)
end

function ITensors.ITensor(o::Ops.Exp, s::IndsNetwork)
  return exp(ITensor(Ops.argument(o), s))
end

function Base.Vector{ITensor}(o::Union{Sum,Prod}, s::IndsNetwork)
  T⃗ = ITensor[]
  for oᵢ in Ops.terms(o)
    Tᵢ = ITensor(oᵢ, s)
    T⃗ = [T⃗; Tᵢ]
  end
  return T⃗
end
