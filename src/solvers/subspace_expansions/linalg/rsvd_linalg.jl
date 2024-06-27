function rsvd_iterative(
    T,
    A::ITensors.ITensorNetworkMaps.ITensorNetworkMap,
    linds::Vector{<:Index};
    theflux=nothing,
    svd_kwargs...,
  )
  
    #translate from in/out to l/r logic
    ininds = ITensors.ITensorNetworkMaps.input_inds(A)
    outinds = ITensors.ITensorNetworkMaps.output_inds(A)
    @assert linds == ininds  ##FIXME: this is specific to the way we wrote the subspace expansion, should be fixed in another iteration
    rinds = outinds
  
    CL = combiner(linds...)
    CR = combiner(rinds...)
    cL = uniqueind(inds(CL), linds)
    cR = uniqueind(inds(CR), rinds)
  
    l = CL * A.itensors[1]
    r = A.itensors[end] * CR
  
    if length(A.itensors) !== 2
      AC = ITensors.ITensorNetworkMaps.ITensorNetworkMap(
        [l, A.itensors[2:(length(A.itensors) - 1)]..., r],
        commoninds(l, CL),
        commoninds(r, CR),
      )
    else
      AC = ITensors.ITensorNetworkMaps.ITensorNetworkMap(
        [l, r], commoninds(l, CL), commoninds(r, CR)
      )
    end
    ###this initializer part is still a bit ugly
    n_init = 1
    p_rule(n) = 2 * n
    ndict2 = init_guess_sizes(cR, n_init, p_rule; theflux=theflux)
    ndict = init_guess_sizes(cL, n_init, p_rule; theflux=theflux)
    if hasqns(ininds)
      ndict = merge(ndict, ndict2)
    else
      ndict = ndict2
    end
    M = build_guess_matrix(T, cR, ndict; theflux=theflux)
    fact, Q = rsvd_core(AC, M; svd_kwargs...)
    n_inc = 2
    ndict = increment_guess_sizes(ndict, n_inc, p_rule)
    new_fact = deepcopy(fact)
    while true
      M = build_guess_matrix(T, cR, ndict; theflux=theflux)
      new_fact, Q = rsvd_core(AC, M; svd_kwargs...)
      if is_converged!(ndict, fact, new_fact; n_inc, has_qns=hasqns(ininds), svd_kwargs...)
        break
      else
        fact = new_fact
      end
    end
    vals = diag(array(new_fact.S))
    (length(vals) == 1 && vals[1]^2 ≤ get(svd_kwargs, :cutoff, 0.0)) &&
      return nothing, nothing, nothing
    return dag(CL) * Q * new_fact.U, new_fact.S, new_fact.V * dag(CR)
  end
  
  function rsvd_iterative(A::ITensor, linds::Vector{<:Index}; svd_kwargs...)
    rinds = uniqueinds(A, linds)
    CL = combiner(linds)
    CR = combiner(rinds)
    AC = CL * A * CR
    cL = combinedind(CL)
    cR = combinedind(CR)
    if inds(AC) != (cL, cR)
      AC = permute(AC, cL, cR)
    end
    n_init = 1
    p_rule(n) = 2 * n
    iszero(norm(AC)) && return nothing, nothing, nothing
    #@show flux(AC)
    ndict2 = init_guess_sizes(cR, n_init, p_rule; theflux=hasqns(AC) ? flux(AC) : nothing)
    ndict = init_guess_sizes(cL, n_init, p_rule; theflux=hasqns(AC) ? flux(AC) : nothing)
    ndict = merge(ndict, ndict2)
    M = build_guess_matrix(eltype(AC), cR, ndict; theflux=hasqns(AC) ? flux(AC) : nothing)
    fact, Q = rsvd_core(AC, M; svd_kwargs...)
    n_inc = 1
    ndict = increment_guess_sizes(ndict, n_inc, p_rule)
    new_fact = deepcopy(fact)
    while true
      M = build_guess_matrix(eltype(AC), cR, ndict; theflux=hasqns(AC) ? flux(AC) : nothing)
      new_fact, Q = rsvd_core(AC, M; svd_kwargs...)
      if is_converged!(ndict, fact, new_fact; n_inc, has_qns=hasqns(AC), svd_kwargs...)
        break
      else
        fact = new_fact
      end
    end
    vals = diag(array(new_fact.S))
    (length(vals) == 1 && vals[1]^2 ≤ get(svd_kwargs, :cutoff, 0.0)) &&
      return nothing, nothing, nothing
    #@show flux(dag(CL)*Q*new_fact.U)
    #@show flux(new_fact.S)\
    
    @assert flux(new_fact.S) == flux(AC)
    return dag(CL) * Q * new_fact.U, new_fact.S, new_fact.V * dag(CR)
    #ToDo?: handle non-QN case separately because there it is advisable to start with n_init closer to target maxdim_expand
    ##not really an issue anymore since we do *2 increase, so only log number of calls
  end

  function rsvd_iterative(A::Vector{ITensor}, linds::Vector{<:Index}; svd_kwargs...)
    if length(A)==1
        rinds = uniqueinds(only(A), linds)
    else
        rinds = uniqueinds(A[end],unioninds(A[1:end-1]...))
    end
    CL = combiner(linds)
    CR = combiner(rinds)
    AC=copy(A)
    AC[1] = CL*first(AC)
    AC[end] = last(AC)*CR
    cL = combinedind(CL)
    cR = combinedind(CR)
    theflux = any(hasqns.(AC)) ? reduce(+,flux.(AC)) : nothing
    #@show theflux
    n_init = 1
    p_rule(n) = 2 * n
    iszero(norm(AC)) && return nothing, nothing, nothing
    #@show flux(AC)
    ndict2 = init_guess_sizes(cR, n_init, p_rule; theflux)
    #@assert isempty(setdiff(keys(Dict(space(cR))), keys(ndict2)))
    ndict = init_guess_sizes(cL, n_init, p_rule; theflux)
    #FIXME: merging is not the right thing to do, but is a workaround due to the way is_converged! is implemented
    ndict = merge(ndict, ndict2)    
    #@show keys(ndict), keys(Dict(space(cR)))
    #@assert isempty(setdiff(keys(Dict(space(cR))), keys(ndict)))
    #@assert isempty(setdiff(keys(ndict),keys(Dict(space(cR)))))
    
    M = build_guess_matrix(eltype(first(AC)), cR, ndict; theflux)
    fact, Q = rsvd_core(AC, M; svd_kwargs...)
    n_inc = 1
    ndict = increment_guess_sizes(ndict, n_inc, p_rule)
    new_fact = deepcopy(fact)
    while true
      M = build_guess_matrix(eltype(first(AC)), cR, ndict; theflux)
      new_fact, Q = rsvd_core(AC, M; svd_kwargs...)
      isnothing(Q) && return nothing,nothing,nothing
      if is_converged!(ndict, fact, new_fact; n_inc, has_qns=any(hasqns.(AC)), svd_kwargs...)
        break
      else
        fact = new_fact
      end
    end
    vals = diag(array(new_fact.S))
    (length(vals) == 1 && vals[1]^2 ≤ get(svd_kwargs, :cutoff, 0.0)) &&
      return nothing, nothing, nothing
    #@show flux(dag(CL)*Q*new_fact.U)
    #@show flux(new_fact.S), theflux
    #@assert flux(new_fact.S) == theflux
    #@show inds(new_fact.U)
    #@show inds(Q)
    #@show inds(dag(CL))
    #@show inds(new_fact.S)
    #@show inds(new_fact.V)
    #@show inds(dag(CR))
    
    return dag(CL) * Q * new_fact.U, new_fact.S, new_fact.V * dag(CR)
    #ToDo?: handle non-QN case separately because there it is advisable to start with n_init closer to target maxdim_expand
    ##not really an issue anymore since we do *2 increase, so only log number of calls
  end

  

  
  function rsvd(A::ITensor, linds::Vector{<:Index}, n::Int, p::Int; svd_kwargs...)
    rinds = uniqueinds(A, linds)
    #ToDo handle empty rinds
    #boilerplate matricization of tensor for matrix decomp
    CL = combiner(linds)
    CR = combiner(rinds)
    AC = CL * A * CR
    cL = combinedind(CL)
    cR = combinedind(CR)
    if inds(AC) != (cL, cR)
      AC = permute(AC, cL, cR)
    end
    M = build_guess_matrix(eltype(AC), cR, n, p; theflux=hasqns(AC) ? flux(AC) : nothing)
    fact, Q = rsvd_core(AC, M; svd_kwargs...)
    return dag(CL) * Q * fact.U, fact.S, fact.V * dag(CR)
  end
  
  function rsvd(A::Vector{<:ITensor}, linds::Vector{<:Index}, n::Int, p::Int; svd_kwargs...)
    if length(A)==1
        rinds = uniqueinds(only(A), linds)
    else
        rinds = uniqueinds(A[end],A[end-1])
    end
    #ToDo handle empty rinds
    #boilerplate matricization of tensor for matrix decomp
    CL = combiner(linds)
    CR = combiner(rinds)
    @assert !isnothing(commonind(CL,first(A)))
    @assert !isnothing(commonind(CR,last(A)))
    AC=copy(A)
    AC[1] = CL*first(AC)
    AC[end] = last(AC)*CR
    
    cL = combinedind(CL)
    cR = combinedind(CR)
    theflux = any(hasqns.(AC)) ? reduce(+,flux.(AC)) : nothing
    #theflux = mapreduce(flux,+,AC)   
    M = build_guess_matrix(eltype(first(AC)), cR, n, p; theflux)
    fact, Q = rsvd_core(AC, M; svd_kwargs...)
    return dag(CL) * Q * fact.U, fact.S, fact.V * dag(CR)
  end

  function rsvd_core(AC::ITensor, M; svd_kwargs...)
    Q = AC * M
    #@show dims(Q)
    Q = ITensors.qr(Q, commoninds(AC, Q))[1]
    QAC = dag(Q) * AC
    fact = svd(QAC, uniqueind(QAC, AC); svd_kwargs...)
    return fact, Q
  end
  
  function rsvd_core(AC::ITensors.ITensorNetworkMaps.ITensorNetworkMap, M; svd_kwargs...)
    #assumes that we want to do a contraction of M with map over its maps output_inds, i.e. a right-multiply
    #thus a transpose is necessary
    Q = transpose(AC) * M
    Q = ITensors.qr(Q, ITensors.ITensorNetworkMaps.input_inds(AC))[1]
    QAC = AC * dag(Q)
    @assert typeof(QAC) <: ITensor
    #@show inds(QAC)
    #@assert !iszero(norm(QAC))
  
    fact = svd(
      QAC, uniqueind(inds(Q), ITensors.ITensorNetworkMaps.input_inds(AC)); svd_kwargs...
    )
    return fact, Q
  end

function rsvd_core(AC::Vector{ITensor}, M; svd_kwargs...)
    #assumes that we want to do a contraction of M with map over its maps output_inds, i.e. a right-multiply
    #thus a transpose is necessary
    @assert !isnothing(commonind(last(AC),M))
    Q = foldr(*,AC;init=M)
    Q = ITensors.qr(Q, commoninds(Q,first(AC)))[1]
    #@show flux(Q)
    #@show nnzblocks(Q)
    any(isequal(0),dims(Q)) && return nothing, nothing ,nothing ,nothing
    QAC = foldl(*,AC,init=dag(Q))
    #@show inds(QAC)
    #@show inds(Q)
    #@show inds(first(AC))
    #@show inds(last(AC))
    
    @assert typeof(QAC) <: ITensor
    
    fact = svd(
      QAC, commoninds(dag(Q), QAC); svd_kwargs...
    )
    return fact, Q
  end