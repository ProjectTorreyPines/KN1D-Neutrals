const _SIGMAV_H1S_HN_HP_B = [
    -1.670435653561e+1, -6.035644995682e-1, -1.942745783445e-8,
    -2.005952284492e-7,  2.962996104431e-8,  2.134293274971e-8,
    -6.353973401838e-9,  6.152557460831e-10, -2.025361858319e-11,
]

function sigmav_h1s_hn_hp(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_H1S_HN_HP_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_h1s_hn_hp(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_h1s_hn_hp(Te[i])
    end
    return out
end
