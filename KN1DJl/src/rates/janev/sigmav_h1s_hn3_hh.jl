const _SIGMAV_H1S_HN3_HH_B = [
    -3.884976142596e+1, 1.520368281111e+1, -6.078494762845e+0,
     1.535455119900e+0, -2.628667482712e-1, 2.994456451213e-2,
    -2.156175515382e-3, 8.826547202670e-5, -1.558890013181e-6,
]

function sigmav_h1s_hn3_hh(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_H1S_HN3_HH_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_h1s_hn3_hh(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_h1s_hn3_hh(Te[i])
    end
    return out
end
