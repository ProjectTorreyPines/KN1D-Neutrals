const _SIGMA_EL_P_H_LOW = [
    -3.233966e1, -1.126918e-1,  5.287706e-3, -2.445017e-3,
    -1.044156e-3,  8.419691e-5,  3.824773e-5,
]

const _SIGMA_EL_P_H_HIGH = [
    -3.231141e1, -1.386002e-1,
]

function sigma_el_p_h(E::Real)::Float64
    Ec = clamp(Float64(E), 0.001, 1.01e5)
    coeffs = Ec < 10.0 ? _SIGMA_EL_P_H_LOW : _SIGMA_EL_P_H_HIGH
    return exp(poly(log(Ec), coeffs)) * 1e-4
end

function sigma_el_p_h(E::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(E))
    @inbounds for i in eachindex(E, out)
        out[i] = sigma_el_p_h(E[i])
    end
    return out
end
