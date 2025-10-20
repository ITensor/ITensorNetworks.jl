function runge_kutta_solver(H, t, ψ; order = 4, kws...)
    # For linear ODE, Runge-Kutta is a Taylor series.
    # Pattern below derived as:
    # exp(tH)ψ = ψ + tHψ +  (tH)^2(ψ)/2! + (tH)^3(ψ)/3! + ...
    # = ψ + (tH) * (ψ + (tH)/2 * (ψ + (tH)/3 * (ψ + ...)))
    eHψ = copy(ψ)
    for ord in reverse(1:order)
        eHψ = (t / ord) * H(eHψ) + ψ
    end
    return eHψ, (;)
end
