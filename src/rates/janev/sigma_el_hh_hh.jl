const _SIGMA_EL_HH_HH_ALPHA = [
    -3.430345e1, -2.960406e-1, -6.382532e-2, -7.557519e-3, 2.606259e-4,
]

function sigma_el_hh_hh(E::Real; vis::Bool=false)::Float64
    vis && @warn "WARNING in SIGMA_EL_HH_HH => using momentum transfer as viscosity cross-section" maxlog=1
    return _eval_log_poly_rate(E, _SIGMA_EL_HH_HH_ALPHA, 0.03, 1.01e4, 1e-4)
end

function sigma_el_hh_hh(E::AbstractVector{<:Real}; vis::Bool=false)::Vector{Float64}
    vis && @warn "WARNING in SIGMA_EL_HH_HH => using momentum transfer as viscosity cross-section" maxlog=1
    out = Vector{Float64}(undef, length(E))
    @inbounds for i in eachindex(E, out)
        out[i] = sigma_el_hh_hh(E[i])
    end
    return out
end
