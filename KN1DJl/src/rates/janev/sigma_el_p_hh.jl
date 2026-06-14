const _SIGMA_EL_P_HH_ALPHA = [
    -3.355719e1, -5.696568e-1, -4.089556e-2, -1.143513e-2, 5.926596e-4,
]

function sigma_el_p_hh(E::Real)::Float64
    return _eval_log_poly_rate(E, _SIGMA_EL_P_HH_ALPHA, 0.03, 1.01e4, 1e-4)
end

function sigma_el_p_hh(E::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(E))
    @inbounds for i in eachindex(E, out)
        out[i] = sigma_el_p_hh(E[i])
    end
    return out
end
