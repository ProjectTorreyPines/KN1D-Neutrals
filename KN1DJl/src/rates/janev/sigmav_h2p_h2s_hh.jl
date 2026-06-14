const _SIGMAV_H2P_H2S_HH_B = [
    -4.794288960529e+1, 2.629649351119e+1, -1.151117702256e+1,
     2.991954880790e+0, -4.949305181578e-1, 5.236320848415e-2,
    -3.433774290547e-3, 1.272097387363e-4, -2.036079507592e-6,
]

function sigmav_h2p_h2s_hh(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_H2P_H2S_HH_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_h2p_h2s_hh(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_h2p_h2s_hh(Te[i])
    end
    return out
end
