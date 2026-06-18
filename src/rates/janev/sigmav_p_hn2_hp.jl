const _SIGMAV_P_HN2_HP_B = [
    -3.408905929046e+1, 1.573560727511e+1, -6.992177456733e+0,
     1.852216261706e+0, -3.130312806531e-1, 3.383704123189e-2,
    -2.265770525273e-3, 8.565603779673e-5, -1.398131377085e-6,
]

function sigmav_p_hn2_hp(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_P_HN2_HP_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_p_hn2_hp(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_p_hn2_hp(Te[i])
    end
    return out
end
