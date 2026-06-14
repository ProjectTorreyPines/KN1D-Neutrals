const _SIGMA_EL_H_HH_ALPHA = [
    -3.495671e1, -4.062257e-1, -3.820531e-2, -9.404486e-3, 3.963723e-4,
]

function sigma_el_h_hh(E::Real)::Float64
    return _eval_log_poly_rate(E, _SIGMA_EL_H_HH_ALPHA, 0.03, 1.01e4, 1e-4)
end

function sigma_el_h_hh(E::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(E))
    @inbounds for i in eachindex(E, out)
        out[i] = sigma_el_h_hh(E[i])
    end
    return out
end
