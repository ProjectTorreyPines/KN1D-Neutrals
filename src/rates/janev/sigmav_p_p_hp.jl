const _SIGMAV_P_P_HP_B = [
    -3.746192301092e+1, 1.559355031108e+1, -6.693238367093e+0,
     1.981700292134e+0, -4.044820889297e-1, 5.352391623039e-2,
    -4.317451841436e-3, 1.918499873454e-4, -3.591779705419e-6,
]

function sigmav_p_p_hp(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_P_P_HP_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_p_p_hp(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_p_p_hp(Te[i])
    end
    return out
end
