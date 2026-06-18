const _SIGMA_CX_HH_ALPHA = [
    -3.427958758517e+01, -7.121484125189e-02,  4.690466187943e-02,
    -8.033946660540e-03, -2.265090924593e-03, -2.102414848737e-04,
     1.948869487515e-04, -2.208124950005e-05,  7.262446915488e-07,
]

function sigma_cx_hh(E::Real)::Float64
    return _eval_log_poly_rate(E, _SIGMA_CX_HH_ALPHA, 0.1, 2.01e4, 1e-4)
end

function sigma_cx_hh(E::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(E))
    @inbounds for i in eachindex(E, out)
        out[i] = sigma_cx_hh(E[i])
    end
    return out
end
