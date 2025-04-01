using .BaseExtensions: maybe_real
using ITensorBase: ITensor, hascommoninds
using QuantumOperatorAlgebra: Applied, Exp, Op, Prod, Scaled, Sum, op
using .ITensorsExtensions: tensor_sum

function ITensorBase.ITensor(o::Op, s::IndsNetwork)
  s⃗ = [only(s[nᵢ]) for nᵢ in Ops.sites(o)]
  return op(Ops.which_op(o), s⃗; Ops.params(o)...)
end

function ITensorBase.ITensor(∏o::Prod, s::IndsNetwork)
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

function ITensorBase.ITensor(∑o::Sum, s::IndsNetwork)
  T = ITensor(0)
  for oᵢ in Ops.terms(∑o)
    Tᵢ = ITensor(oᵢ, s)
    T = tensor_sum(T, Tᵢ)
  end
  return T
end

function ITensorBase.ITensor(o::Scaled, s::IndsNetwork)
  return maybe_real(Ops.coefficient(o)) * ITensor(Ops.argument(o), s)
end

function ITensorBase.ITensor(o::Exp, s::IndsNetwork)
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
