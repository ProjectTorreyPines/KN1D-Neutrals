const _SIGMA_CX_H0_ALPHA = [
    -3.274123792568e+01, -8.916456579806e-02, -3.016990732025e-02,
     9.205482406462e-03,  2.400266568315e-03, -1.927122311323e-03,
     3.654750340106e-04, -2.788866460622e-05,  7.422296363524e-07,
]

function sigma_cx_h0(E::Real; freeman::Bool=false)::Float64
    Ec = freeman ? clamp(Float64(E), 0.1, 1e5) : clamp(Float64(E), 0.1, 2.01e4)
    if freeman
        return 1.0e-4 * 0.6937e-14 * (1.0 - 0.155 * log10(Ec))^2 / (1.0 + 0.1112e-14 * Ec^3.3)
    end
    return exp(poly(log(Ec), _SIGMA_CX_H0_ALPHA)) * 1e-4
end

function sigma_cx_h0(E::AbstractVector{<:Real}; freeman::Bool=false)::Vector{Float64}
    out = Vector{Float64}(undef, length(E))
    @inbounds for i in eachindex(E, out)
        out[i] = sigma_cx_h0(E[i]; freeman=freeman)
    end
    return out
end
