const _SIGMAV_H1S_H2S_HH_B = [
    -3.454175591367e+1, 1.412655911280e+1, -6.004466156761e+0,
     1.589476697488e+0, -2.775796909649e-1, 3.152736888124e-2,
    -2.229578042005e-3, 8.890114963166e-5, -1.523912962346e-6,
]

function sigmav_h1s_h2s_hh(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_H1S_H2S_HH_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_h1s_h2s_hh(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_h1s_h2s_hh(Te[i])
    end
    return out
end
