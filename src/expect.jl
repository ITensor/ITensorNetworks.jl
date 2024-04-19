using ITensors.ITensorMPS: ITensorMPS, expect, promote_itensor_eltype, OpSum

function ITensorMPS.expect(
  op::String,
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=false,
  sequence=nothing,
  vertices=vertices(ψ),
)
  s = siteinds(ψ)
  ElT = promote_itensor_eltype(ψ)
  # ElT = ishermitian(ITensors.op(op, s[vertices[1]])) ? real(ElT) : ElT
  res = Dictionary(vertices, Vector{ElT}(undef, length(vertices)))
  if isnothing(sequence)
    sequence = contraction_sequence(inner_network(ψ, ψ))
  end
  normψ² = norm_sqr(ψ; alg="exact", sequence)
  for v in vertices
    O = ITensor(Op(op, v), s)
    Oψ = apply(O, ψ; cutoff, maxdim, ortho)
    res[v] = inner(ψ, Oψ; alg="exact", sequence) / normψ²
  end
  return res
end

function ITensorMPS.expect(
  ℋ::OpSum,
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=false,
  sequence=nothing,
)
  s = siteinds(ψ)
  # h⃗ = Vector{ITensor}(ℋ, s)
  if isnothing(sequence)
    sequence = contraction_sequence(inner_network(ψ, ψ))
  end
  h⃗ψ = [apply(hᵢ, ψ; cutoff, maxdim, ortho) for hᵢ in ITensors.terms(ℋ)]
  ψhᵢψ = [inner(ψ, hᵢψ; alg="exact", sequence) for hᵢψ in h⃗ψ]
  ψh⃗ψ = sum(ψhᵢψ)
  ψψ = norm_sqr(ψ; alg="exact", sequence)
  return ψh⃗ψ / ψψ
end

function ITensorMPS.expect(
  opsum_sum::Sum{<:OpSum},
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=true,
  sequence=nothing,
)
  return expect(sum(Ops.terms(opsum_sum)), ψ; cutoff, maxdim, ortho, sequence)
end
