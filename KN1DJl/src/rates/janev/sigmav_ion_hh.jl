const _SIGMAV_ION_HH_B = [
    -3.568640293666e+1, 1.733468989961e+1, -7.767469363538e+0,
     2.211579405415e+0, -4.169840174384e-1, 5.088289820867e-2,
    -3.832737518325e-3, 1.612863120371e-4, -2.893391904431e-6,
]

function sigmav_ion_hh(Te::Real)::Float64
    return _eval_log_poly_rate(Te, _SIGMAV_ION_HH_B, 0.1, 2.01e4, 1e-6)
end

function sigmav_ion_hh(Te::AbstractVector{<:Real})::Vector{Float64}
    out = Vector{Float64}(undef, length(Te))
    @inbounds for i in eachindex(Te, out)
        out[i] = sigmav_ion_hh(Te[i])
    end
    return out
end
