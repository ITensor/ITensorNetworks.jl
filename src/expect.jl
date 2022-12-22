function expect(
  op::String,
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=false,
  sequence=nothing,
  sites=vertices(ψ),
)
  s = siteinds(ψ)
  ElT = promote_itensor_eltype(ψ)
  # ElT = ishermitian(ITensors.op(op, s[sites[1]])) ? real(ElT) : ElT
  res = Dictionary(sites, Vector{ElT}(undef, length(sites)))
  if isnothing(sequence)
    sequence = contraction_sequence(inner_network(ψ, ψ; flatten=true))
  end
  normψ² = norm_sqr(ψ; sequence)
  for v in sites
    O = ITensor(Op(op, v), s)
    Oψ = apply(O, ψ; cutoff, maxdim, ortho)
    res[v] = contract_inner(ψ, Oψ; sequence) / normψ²
  end
  return res
end

function expect(
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

function expect(
  opsum_sum::Sum{<:OpSum},
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  ortho=true,
  sequence=nothing,
)
  return expect(sum(Ops.terms(opsum_sum)), ψ; cutoff, maxdim, ortho, sequence)
end
