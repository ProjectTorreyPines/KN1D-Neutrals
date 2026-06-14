const _SIGMAV_P_H1S_HP_B = [
    -1.781416067709e+1, 2.277799785711e+0, -1.266868411626e+0,
     4.296170447419e-1, -9.609908013189e-2, 1.387958040699e-2,
    -1.231349039470e-3, 6.042383126281e-5, -1.247521040900e-6,
]

function sigmav_p_h1s_hp(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_P_H1S_HP_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_p_h1s_hp(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_p_h1s_hp(Te[i])
    end
    return out
end
