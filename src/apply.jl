function ITensors.apply(
  o::ITensor,
  ψ::AbstractITensorNetwork;
  cutoff=nothing,
  maxdim=nothing,
  normalize=false,
  ortho=false,
)
  ψ = copy(ψ)
  v⃗ = neighbor_vertices(ψ, o)
  if length(v⃗) == 1
    if ortho
      ψ = orthogonalize(ψ, v⃗[1])
    end
    oψᵥ = apply(o, ψ[v⃗[1]])
    if normalize
      oψᵥ ./= norm(oψᵥ)
    end
    ψ[v⃗[1]] = oψᵥ
  elseif length(v⃗) == 2
    e = v⃗[1] => v⃗[2]
    if !has_edge(ψ, e)
      error("Vertices where the gates are being applied must be neighbors for now.")
    end
    if ortho
      ψ = orthogonalize(ψ, v⃗[1])
    end
    
    #Check whether to do a memory efficient QR first (do both sites combined for now, although we could split if we wish)
    #if no maxdim is provided just do it based on inds counting, if it is provided do it based on whichever takes less memory
    d1, d2 = prod(dim.(inds(ψ[v⃗[1]], tags ="Site"))), prod(dim.(inds(ψ[v⃗[2]], tags ="Site")))
    χv1, χv2 = prod(dim.(uniqueinds(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), inds(ψ[v⃗[1]], tags ="Site")))), prod(dim.(uniqueinds(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), inds(ψ[v⃗[2]], tags ="Site"))))
    if (maxdim == nothing && length(inds(ψ[v⃗[1]])) + length(inds(ψ[v⃗[2]])) <= 6) || (maxdim != nothing && maxdim*maxdim*d1*d2 >= χv1*χv2)
      oψᵥ = apply(o, ψ[v⃗[1]] * ψ[v⃗[2]])
      ψᵥ₁, ψᵥ₂ = factorize(
        oψᵥ, inds(ψ[v⃗[1]]); cutoff, maxdim, tags=ITensorNetworks.edge_tag(e)
      )
    else
      Qv1, Rv1 = factorize(ψ[v⃗[1]], uniqueinds(uniqueinds(ψ[v⃗[1]], ψ[v⃗[2]]), inds(ψ[v⃗[1]], tags ="Site")))  
      Qv2, Rv2 = factorize(ψ[v⃗[2]], uniqueinds(uniqueinds(ψ[v⃗[2]], ψ[v⃗[1]]), inds(ψ[v⃗[2]], tags ="Site")))  
      Rv1_new, Rv2_new = factorize(noprime(Rv1*o*Rv2), inds(Rv1); cutoff, maxdim, tags=ITensorNetworks.edge_tag(e))  
      ψᵥ₁ = Qv1 * Rv1_new  
      ψᵥ₂ = Qv2 * Rv2_new
    end

    if normalize
      ψᵥ₁ ./= norm(ψᵥ₁)
      ψᵥ₂ ./= norm(ψᵥ₂)
    end

    ψ[v⃗[1]] = ψᵥ₁
    ψ[v⃗[2]] = ψᵥ₂

  elseif length(v⃗) < 1
    error("Gate being applied does not share indices with tensor network.")
  elseif length(v⃗) > 2
    error("Gates with more than 2 sites is not supported yet.")
  end
  return ψ
end

function ITensors.apply(
  o⃗::Vector{ITensor},
  ψ::AbstractITensorNetwork;
  cutoff,
  maxdim=typemax(Int),
  normalize=false,
  ortho=false,
)
  o⃗ψ = ψ
  for oᵢ in o⃗
    o⃗ψ = apply(oᵢ, o⃗ψ; cutoff, maxdim, normalize, ortho)
  end
  return o⃗ψ
end

function ITensors.apply(
  o⃗::Scaled, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  return maybe_real(Ops.coefficient(o⃗)) *
         apply(Ops.argument(o⃗), ψ; cutoff, maxdim, normalize, ortho)
end

function ITensors.apply(
  o⃗::Prod, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  o⃗ψ = ψ
  for oᵢ in o⃗
    o⃗ψ = apply(oᵢ, o⃗ψ; cutoff, maxdim, normalize, ortho)
  end
  return o⃗ψ
end

function ITensors.apply(
  o::Op, ψ::AbstractITensorNetwork; cutoff, maxdim, normalize=false, ortho=false
)
  return apply(ITensor(o, siteinds(ψ)), ψ; cutoff, maxdim, normalize, ortho)
end