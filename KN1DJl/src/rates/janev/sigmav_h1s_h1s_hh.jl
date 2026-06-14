const _SIGMAV_H1S_H1S_HH_B = [
    -2.787217511174e+1, 1.052252660075e+1, -4.973212347860e+0,
     1.451198183114e+0, -3.062790554644e-1, 4.433379509258e-2,
    -4.096344172875e-3, 2.159670289222e-4, -4.928545325189e-6,
]

function sigmav_h1s_h1s_hh(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_H1S_H1S_HH_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_h1s_h1s_hh(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_h1s_h1s_hh(Te[i])
    end
    return out
end
