function _eval_log_poly_rate(
    x::Real,
    coeffs::AbstractVector{<:Real},
    lo::Float64,
    hi::Float64,
    scale::Float64,
)::Float64
    xc = clamp(Float64(x), lo, hi)
    return exp(poly(log(xc), coeffs)) * scale
end

function _eval_bivariate_log_poly_rate(
    T::Real,
    E::Real,
    alpha::AbstractMatrix{<:Real},
    lo::Float64,
    hi::Float64,
    scale::Float64,
)::Float64
    Tc = clamp(Float64(T), lo, hi)
    Ec = clamp(Float64(E), lo, hi)
    logT = log(Tc)
    logE = log(Ec)

    result = 0.0
    Ei = 1.0
    @inbounds for i in axes(alpha, 2)
        Tj = 1.0
        for j in axes(alpha, 1)
            result += Float64(alpha[j, i]) * Ei * Tj
            Tj *= logT
        end
        Ei *= logE
    end

    return exp(result) * scale
end
