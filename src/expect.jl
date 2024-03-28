using ITensors.ITensorMPS: ITensorMPS

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
    sequence = contraction_sequence(inner_network(ψ, ψ; flatten=true))
  end
  normψ² = norm_sqr(ψ; sequence)
  for v in vertices
    O = ITensor(Op(op, v), s)
    Oψ = apply(O, ψ; cutoff, maxdim, ortho)
    res[v] = contract_inner(ψ, Oψ; sequence) / normψ²
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
    sequence = contraction_sequence(inner_network(ψ, ψ; flatten=true))
  end
  h⃗ψ = [apply(hᵢ, ψ; cutoff, maxdim, ortho) for hᵢ in ITensors.terms(ℋ)]
  ψhᵢψ = [contract_inner(ψ, hᵢψ; sequence) for hᵢψ in h⃗ψ]
  ψh⃗ψ = sum(ψhᵢψ)
  ψψ = norm_sqr(ψ; sequence)
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
