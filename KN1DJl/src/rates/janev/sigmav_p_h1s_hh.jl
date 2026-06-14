const _SIGMAV_P_H1S_HH_B = [
    -3.834597006782e+1, 1.426322356722e+1, -5.826468569506e+0,
     1.727940947913e+0, -3.598120866343e-1, 4.822199350494e-2,
    -3.909402993006e-3, 1.738776657690e-4, -3.252844486351e-6,
]

function sigmav_p_h1s_hh(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_P_H1S_HH_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_p_h1s_hh(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_p_h1s_hh(Te[i])
    end
    return out
end
