const _SIGMA_EL_H_H_MOMENTUM = [
    -3.330843e1, -5.738374e-1, -1.028610e-1, -3.920980e-3, 5.964135e-4,
]

const _SIGMA_EL_H_H_VISCOSITY = [
    -3.344860e1, -4.238982e-1, -7.477873e-2, -7.915053e-3, -2.686129e-4,
]

function sigma_el_h_h(E::Real; vis::Bool=false)::Float64
    coeffs = vis ? _SIGMA_EL_H_H_VISCOSITY : _SIGMA_EL_H_H_MOMENTUM
    return _eval_log_poly_rate(E, coeffs, 0.03, 1.01e4, 1e-4)
end

function sigma_el_h_h(E::AbstractVector{<:Real}; vis::Bool=false)::Vector{Float64}
    out = Vector{Float64}(undef, length(E))
    @inbounds for i in eachindex(E, out)
        out[i] = sigma_el_h_h(E[i]; vis=vis)
    end
    return out
end
