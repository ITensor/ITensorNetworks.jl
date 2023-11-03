#
# To Do:
# - implement assembleLanczosVectors
# - check slice ranges - change end value by 1?
#

function assemble_lanczos_vecs(lanczos_vectors, linear_comb, norm)
  #if length(lanczos_vectors) != length(linear_comb)
  #  @show length(lanczos_vectors)
  #  @show length(linear_comb)
  #end
  xt = norm * linear_comb[1] * lanczos_vectors[1]
  for i in 2:length(lanczos_vectors)
    xt += norm * linear_comb[i] * lanczos_vectors[i]
  end
  return xt
end

struct ApplyExpInfo
  numops::Int
  converged::Int
end

function applyexp(H, tau::Number, x0; maxiter=30, tol=1e-12, outputlevel=0, normcutoff=1e-7)
  # Initialize Lanczos vectors
  v1 = copy(x0)
  nrm = norm(v1)
  v1 /= nrm
  lanczos_vectors = [v1]

  ElT = promote_type(typeof(tau), eltype(x0))

  bigTmat = zeros(ElT, maxiter + 3, maxiter + 3)

  nmatvec = 0

  v0 = nothing
  beta = 0.0
  for iter in 1:maxiter
    tmat_size = iter + 1

    # Matrix-vector multiplication
    w = H(v1)
    nmatvec += 1

    avnorm = norm(w)
    alpha = dot(w, v1)

    bigTmat[iter, iter] = alpha

    w -= alpha * v1
    if iter > 1
      w -= beta * v0
    end
    v0 = copy(v1)

    beta = norm(w)

    # check for Lanczos sequence exhaustion
    if abs(beta) < beta_tol
      # Assemble the time evolved state
      tmat = bigTmat[1:tmat_size, 1:tmat_size]
      tmat_exp = exp(tau * tmat)
      linear_comb = tmat_exp[:, 1]
      xt = assemble_lanczos_vecs(lanczos_vectors, linear_comb, nrm)
      return xt, ApplyExpInfo(nmatvec, 1)
    end

    # update next lanczos vector
    v1 = copy(w)
    v1 /= beta
    push!(lanczos_vectors, v1)
    bigTmat[iter + 1, iter] = beta
    bigTmat[iter, iter + 1] = beta

    # Convergence check
    if iter > 0
      # Prepare extended T-matrix for exponentiation
      tmat_ext_size = tmat_size + 2
      tmat_ext = bigTmat[1:tmat_ext_size, 1:tmat_ext_size]

      tmat_ext[tmat_size - 1, tmat_size] = 0.0
      tmat_ext[tmat_size + 1, tmat_size] = 1.0

      # Exponentiate extended T-matrix
      tmat_ext_exp = exp(tau * tmat_ext)

      ϕ1 = abs(nrm * tmat_ext_exp[tmat_size, 1])
      ϕ2 = abs(nrm * tmat_ext_exp[tmat_size + 1, 1] * avnorm)

      if ϕ1 > 10 * ϕ2
        error = ϕ2
      elseif (ϕ1 > ϕ2)
        error = (ϕ1 * ϕ2) / (ϕ1 - ϕ2)
      else
        error = ϕ1
      end

      if outputlevel >= 3
        @printf("  Iteration: %d, Error: %.2E\n", iter, error)
      end

      if ((error < tol) || (iter == maxiter))
        converged = 1
        if (iter == maxiter)
          println("warning: applyexp not converged in $maxiter steps")
          converged = 0
        end

        # Assemble the time evolved state
        linear_comb = tmat_ext_exp[:, 1]
        xt = assemble_lanczos_vecs(lanczos_vectors, linear_comb, nrm)

        if outputlevel >= 3
          println("  Number of iterations: $iter")
        end

        return xt, ApplyExpInfo(nmatvec, converged)
      end
    end  # end convergence test
  end # iter

  if outputlevel >= 0
    println("In applyexp, number of matrix-vector multiplies: ", nmatvec)
  end

  return x0
end
