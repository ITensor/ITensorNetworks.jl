using KrylovKit: KrylovKit

function eigsolve_solver(
        operator,
        init,
        howmany = 1;
        which_eigval = :SR,
        ishermitian = true,
        tol = 1.0e-14,
        krylovdim = 3,
        maxiter = 1,
        verbosity = 0,
        eager = false,
        kws...
    )
    vals, vecs, info = KrylovKit.eigsolve(
        operator,
        init,
        howmany,
        which_eigval;
        ishermitian,
        tol,
        krylovdim,
        maxiter,
        verbosity,
        eager
    )
    return vals[1], vecs[1]
end
