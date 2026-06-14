struct MaxwellianScratch
    row_integral::Vector{Float64}
    vx_weight::Vector{Float64}
    tmp::Matrix{Float64}
    fmat::Matrix{Float64}
    weighted_dist::Matrix{Float64}
    vx_dist::Matrix{Float64}
    vr_dist::Matrix{Float64}
    vth_dist::Matrix{Float64}
    vth_diffs::Array{Float64,3}
    vrvx_diffs::Array{Float64,3}
    correction::Matrix{Float64}
    out::Matrix{Float64}
end

function MaxwellianScratch(nvr::Int, nvx::Int)::MaxwellianScratch
    return MaxwellianScratch(
        Vector{Float64}(undef, nvr),
        Vector{Float64}(undef, nvx),
        Matrix{Float64}(undef, nvr, nvx),
        Matrix{Float64}(undef, nvr, nvx),
        Matrix{Float64}(undef, nvr + 2, nvx + 2),
        Matrix{Float64}(undef, nvr + 2, nvx + 2),
        Matrix{Float64}(undef, nvr + 2, nvx + 2),
        Matrix{Float64}(undef, nvr + 2, nvx + 2),
        Array{Float64}(undef, nvr, nvx, 2),
        Array{Float64}(undef, nvr, nvx, 2),
        Matrix{Float64}(undef, nvr, nvx),
        Matrix{Float64}(undef, nvr, nvx),
    )
end

@inline function _normalize_slice!(
    f::AbstractMatrix{Float64},
    variable::Vector{Float64},
    vsd::VSpaceDifferentials,
)::Nothing
    nvr, nvx = size(f)
    @inbounds for i in 1:nvr
        row = 0.0
        for j in 1:nvx
            row += f[i, j] * vsd.dvx[j]
        end
        variable[i] = row
    end

    norm = 0.0
    @inbounds for i in 1:nvr
        norm += vsd.dvr_vol[i] * variable[i]
    end

    inv_norm = 1.0 / norm
    @inbounds for j in 1:nvx, i in 1:nvr
        f[i, j] *= inv_norm
    end
    return nothing
end

function compensate_distribution!(
    out::Matrix{Float64},
    scratch::MaxwellianScratch,
    f_slice::AbstractMatrix{<:Real},
    vsd::VSpaceDifferentials,
    vr::AbstractVector{<:Real},
    vx::AbstractVector{<:Real},
    vth::Real,
    target_vx::Real,
    target_energy::Real;
    nb::Real=1.0,
    assume_pos::Bool=true,
)::Float64
    nvr, nvx = size(f_slice)
    length(vr) == nvr || throw(ArgumentError("length(vr) must match size(f_slice, 1)"))
    length(vx) == nvx || throw(ArgumentError("length(vx) must match size(f_slice, 2)"))
    size(out) == (nvr, nvx) || throw(DimensionMismatch("out must match f_slice size"))

    row_integral = scratch.row_integral
    vx_weight = scratch.vx_weight
    tmp = scratch.tmp
    fmat = scratch.fmat
    weighted_dist = scratch.weighted_dist
    vx_dist = scratch.vx_dist
    vr_dist = scratch.vr_dist
    vth_dist = scratch.vth_dist
    vth_diffs = scratch.vth_diffs
    vrvx_diffs = scratch.vrvx_diffs
    correction = scratch.correction

    vthf = Float64(vth)
    nbf = Float64(nb)
    target_vxf = Float64(target_vx)
    target_energyf = Float64(target_energy)

    @inbounds for j in 1:nvx
        vx_weight[j] = Float64(vx[j]) * vsd.dvx[j]
    end

    @inbounds for j in 1:nvx, i in 1:nvr
        fmat[i, j] = Float64(f_slice[i, j])
    end

    mul!(row_integral, fmat, vx_weight)
    vx_moment = vthf * dot(vsd.dvr_vol, row_integral) / nbf

    @inbounds for j in 1:nvx, i in 1:nvr
        tmp[i, j] = vsd.vmag_squared[i, j] * fmat[i, j]
    end
    mul!(row_integral, tmp, vsd.dvx)
    energy_moment = vthf^2 * dot(vsd.dvr_vol, row_integral) / nbf

    fill!(weighted_dist, 0.0)
    @inbounds for j in 1:nvx, i in 1:nvr
        weighted_dist[i + 1, j + 1] = fmat[i, j] * vsd.volume[i, j] / nbf
    end

    allow_neg = false
    if !assume_pos
        cutoff = 1.0e-6 * maximum(weighted_dist)
        @inbounds for j in axes(weighted_dist, 2), i in axes(weighted_dist, 1)
            ax = abs(weighted_dist[i, j])
            if 0.0 < ax < cutoff
                weighted_dist[i, j] = 0.0
            end
        end

        rowmax = -Inf
        @inbounds for j in axes(weighted_dist, 2)
            rowmax = max(rowmax, weighted_dist[3, j])
        end
        allow_neg = rowmax <= 0.0
    end

    @inbounds for j in axes(weighted_dist, 2), i in axes(weighted_dist, 1)
        wd = weighted_dist[i, j]
        vx_dist[i, j] = wd * vsd.vx_dvx[i, j]
        vr_dist[i, j] = wd * vsd.vr_dvr[i, j]
        vth_dist[i, j] = wd * vsd.vth_dvx[i, j]
    end

    fill!(vth_diffs, 0.0)
    @inbounds for j in 1:nvx, i in 1:nvr
        vth_diffs[i, j, 1] = vth_dist[i + 1, j] - vth_dist[i + 1, j + 1]
        vth_diffs[i, j, 2] = -vth_dist[i + 1, j + 2] + vth_dist[i + 1, j + 1]
    end

    pos_start = vsd.vx_pos_start
    pos_end = vsd.vx_pos_end
    neg_start = vsd.vx_neg_start
    neg_end = vsd.vx_neg_end

    fill!(vrvx_diffs, 0.0)
    @inbounds for i in 1:nvr
        ip = i + 1

        for j in pos_start + 1:pos_end
            vrvx_diffs[i, j, 1] = vx_dist[ip, j] - vx_dist[ip, j + 1]
        end
        vrvx_diffs[i, pos_start, 1] = -vx_dist[ip, pos_start + 1]
        vrvx_diffs[i, neg_end, 1] = vx_dist[ip, neg_end + 1]
        for j in neg_start:neg_end - 1
            jp = j + 1
            vrvx_diffs[i, j, 1] = -vx_dist[ip, jp + 1] + vx_dist[ip, jp]
        end
        for j in 1:nvx
            jp = j + 1
            vrvx_diffs[i, j, 1] += vr_dist[i, jp] - vr_dist[ip, jp]
        end
    end

    @inbounds for i in 1:nvr
        ip = i + 1

        for j in pos_start + 1:pos_end
            vrvx_diffs[i, j, 2] = -vx_dist[ip, j + 2] + vx_dist[ip, j + 1]
        end
        vrvx_diffs[i, pos_start, 2] = -vx_dist[ip, pos_start + 2]
        vrvx_diffs[i, neg_end, 2] = vx_dist[ip, neg_end]
        for j in neg_start:neg_end - 1
            vrvx_diffs[i, j, 2] = vx_dist[ip, j] - vx_dist[ip, j + 1]
        end
    end

    @inbounds for i in 2:nvr, j in 1:nvx
        jp = j + 1
        vrvx_diffs[i, j, 2] += vr_dist[i + 1, jp] - vr_dist[i + 2, jp]
    end
    @inbounds for j in 1:nvx
        jp = j + 1
        vrvx_diffs[1, j, 2] -= vr_dist[3, jp]
    end

    if allow_neg
        @inbounds for j in 1:nvx
            jp = j + 1
            vrvx_diffs[1, j, 2] -= vr_dist[2, jp]
            if nvr >= 2
                vrvx_diffs[2, j, 2] += vr_dist[2, jp]
            end
        end
    end

    TB1 = (0.0, 0.0)
    TB2 = (0.0, 0.0)
    signs = (1.0, -1.0)

    ia_final = 1
    ib_final = 1
    vth_scalar = 0.0
    vrvx_scalar = 0.0
    found = false

    @inbounds for ia in 1:2
        TA1 = 0.0
        for i in 1:nvr
            row = 0.0
            for j in 1:nvx
                row += vth_diffs[i, j, ia] * Float64(vx[j])
            end
            TA1 += row
        end
        TA1 *= vthf

        TA2 = 0.0
        for j in 1:nvx, i in 1:nvr
            TA2 += vsd.vmag_squared[i, j] * vth_diffs[i, j, ia]
        end
        TA2 *= vthf^2

        tb1_1, tb1_2 = TB1
        tb2_1, tb2_2 = TB2

        for ib in 1:2
            tb1 = ib == 1 ? tb1_1 : tb1_2
            if tb1 == 0.0
                tmp1 = 0.0
                for i in 1:nvr
                    row = 0.0
                    for j in 1:nvx
                        row += vrvx_diffs[i, j, ib] * Float64(vx[j])
                    end
                    tmp1 += row
                end
                tb1 = vthf * tmp1
                if ib == 1
                    tb1_1 = tb1
                else
                    tb1_2 = tb1
                end
            end

            tb2 = ib == 1 ? tb2_1 : tb2_2
            if tb2 == 0.0
                tmp2 = 0.0
                for j in 1:nvx, i in 1:nvr
                    tmp2 += vsd.vmag_squared[i, j] * vrvx_diffs[i, j, ib]
                end
                tb2 = vthf^2 * tmp2
                if ib == 1
                    tb2_1 = tb2
                else
                    tb2_2 = tb2
                end
            end

            denom = TA2 * tb1 - TA1 * tb2
            local_vrvx_scalar = 0.0
            local_vth_scalar = 0.0
            if denom != 0.0 && TA1 != 0.0
                local_vrvx_scalar =
                    (TA2 * (target_vxf - vx_moment) - TA1 * (target_energyf - energy_moment)) / denom
                local_vth_scalar = (target_vxf - vx_moment - tb1 * local_vrvx_scalar) / TA1
            end

            ia_final = ia
            ib_final = ib
            vth_scalar = local_vth_scalar
            vrvx_scalar = local_vrvx_scalar

            if local_vth_scalar * signs[ia] > 0.0 && local_vrvx_scalar * signs[ib] > 0.0
                found = true
                break
            end
        end

        TB1 = (tb1_1, tb1_2)
        TB2 = (tb2_1, tb2_2)
        found && break
    end

    @inbounds for j in 1:nvx, i in 1:nvr
        correction[i, j] =
            vth_scalar * vth_diffs[i, j, ia_final] +
            vrvx_scalar * vrvx_diffs[i, j, ib_final]
    end

    s = 1.0
    if !assume_pos && !allow_neg
        @inbounds for j in 1:nvx, i in 1:nvr
            wd = weighted_dist[i + 1, j + 1]
            c = correction[i, j]
            if wd > 0.0 && c < 0.0
                s = min(s, -wd / c)
            end
        end
    end

    @inbounds for j in 1:nvx, i in 1:nvr
        wd = weighted_dist[i + 1, j + 1]
        out[i, j] = nbf * (wd + s * correction[i, j]) / vsd.volume[i, j]
    end

    return s
end

function compensate_distribution(
    f_slice::AbstractMatrix{<:Real},
    vsd::VSpaceDifferentials,
    vr::AbstractVector{<:Real},
    vx::AbstractVector{<:Real},
    vth::Real,
    target_vx::Real,
    target_energy::Real;
    nb::Real=1.0,
    assume_pos::Bool=true,
)::Tuple{Matrix{Float64}, Float64}
    nvr, nvx = size(f_slice)
    scratch = MaxwellianScratch(nvr, nvx)
    s = compensate_distribution!(
        scratch.out,
        scratch,
        f_slice,
        vsd,
        vr,
        vx,
        vth,
        target_vx,
        target_energy;
        nb=nb,
        assume_pos=assume_pos,
    )
    return scratch.out, s
end

function create_shifted_maxwellian(
    vr::Vector{Float64},
    vx::Vector{Float64},
    Tmaxwell::Vector{Float64},
    vx_shift::Vector{Float64},
    mu::Int,
    mol::Int,
    Tnorm::Float64,
)::Array{Float64,3}
    nvr = length(vr)
    nvx = length(vx)
    nk = length(vx_shift)
    length(Tmaxwell) == nk || throw(ArgumentError("Tmaxwell and vx_shift must have same length"))

    maxwell = zeros(Float64, nvr, nvx, nk)
    vth = sqrt(2.0 * Q * Tnorm / (mu * H_MASS))
    vsd = VSpaceDifferentials(vr, vx)
    scratches = [MaxwellianScratch(nvr, nvx) for _ in 1:Threads.nthreads()]
    variables = [Vector{Float64}(undef, nvr) for _ in 1:Threads.nthreads()]

    Threads.@threads for k in 1:nk
        Tmaxwell[k] <= 0.0 && continue
        tid = Threads.threadid()
        scratch = scratches[tid]
        variable = variables[tid]

        scale = mol * Tnorm / Tmaxwell[k]
        shift = vx_shift[k] / vth
        @inbounds for j in 1:nvx, i in 1:nvr
            arg = -((vr[i]^2 + (vx[j] - shift)^2) * scale)
            maxwell[i, j, k] = exp((-80.0 < arg < 0.0) ? arg : -80.0)
        end

        fk = @view maxwell[:, :, k]
        _normalize_slice!(fk, variable, vsd)

        target_energy = vx_shift[k]^2 + (3.0 * Q * Tmaxwell[k] / (mol * mu * H_MASS))
        compensate_distribution!(
            scratch.out,
            scratch,
            fk,
            vsd,
            vr,
            vx,
            vth,
            vx_shift[k],
            target_energy,
        )
        fk .= scratch.out

        _normalize_slice!(fk, variable, vsd)
    end

    return maxwell
end
