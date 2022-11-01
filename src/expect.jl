function ITensors.expect(op::String, ψ::AbstractITensorNetwork; cutoff=nothing, maxdim=nothing, ortho=false, sequence=nothing)
  s = siteinds(ψ)
  res = Dictionary(vertices(ψ), Vector{Float64}(undef, nv(ψ)))
  if isnothing(sequence)
    sequence = optimal_contraction_sequence(flattened_inner_network(ψ, ψ))
  end
  normψ² = norm2(ψ; sequence)
  for v in vertices(ψ)
    O = ITensor(Op(op, v), s)
    Oψ = apply(O, ψ; cutoff, maxdim, ortho)
    res[v] = contract_inner(ψ, Oψ; sequence) / normψ²
  end
  return res
end

function ITensors.expect(ℋ::OpSum, ψ::AbstractITensorNetwork; cutoff=nothing, maxdim=nothing, ortho=false, sequence=nothing)
  s = siteinds(ψ)
  # h⃗ = Vector{ITensor}(ℋ, s)
  if isnothing(sequence)
    sequence = optimal_contraction_sequence(flattened_inner_network(ψ, ψ))
  end
  h⃗ψ = [apply(hᵢ, ψ; cutoff, maxdim, ortho) for hᵢ in ITensors.terms(ℋ)]
  ψhᵢψ = [contract_inner(ψ, hᵢψ; sequence) for hᵢψ in h⃗ψ]
  ψh⃗ψ = sum(ψhᵢψ)
  ψψ = norm2(ψ; sequence)
  return ψh⃗ψ / ψψ
end

function ITensors.expect(opsum_sum::Sum{<:OpSum}, ψ::AbstractITensorNetwork; cutoff=nothing, maxdim=nothing, ortho=true, sequence=nothing)
  return expect(sum(Ops.terms(opsum_sum)), ψ; cutoff, maxdim, ortho, sequence)
end
