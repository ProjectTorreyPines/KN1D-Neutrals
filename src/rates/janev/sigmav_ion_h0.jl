const _SIGMAV_ION_H0_B = [
    -3.271396786375e+1, 1.353655609057e+1, -5.739328757388e+0,
     1.563154982022e+0, -2.877056004391e-1, 3.482559773737e-2,
    -2.631976175590e-3, 1.119543953861e-4, -2.039149852002e-6,
]

function sigmav_ion_h0(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_ION_H0_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_ion_h0(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_ion_h0(Te[i])
    end
    return out
end
