function _get_interpolation_bounds(
    a::AbstractVector{<:Real},
    b::AbstractVector{<:Real};
    a_name::String="a",
    b_name::String="b",
)::Bound
    amin, amax = extrema(a)
    first_idx = 0
    last_idx = 0

    @inbounds for i in eachindex(b)
        if amin <= b[i] <= amax
            first_idx == 0 && (first_idx = i)
            last_idx = i
        end
    end

    first_idx == 0 && error("No values of $b_name are within range of $a_name")
    return Bound(first_idx, last_idx)
end

@inline function _locate_python_style(table::AbstractVector{<:Real}, value::Real)::Int
    idx0 = searchsortedlast(table, value) - 1
    if value < table[begin]
        return -1
    elseif value >= table[end]
        return length(table) - 1
    end
    return idx0
end

function _test_bounds(
    fb::AbstractArray{T,3},
    test_bound::Bound,
    var_len::Integer,
    test_axis::Integer,
    iter_bound1::Bound,
    iter_bound2::Bound,
    do_warn::Real,
    var_name::String="a",
)::Nothing where {T<:Real}
    big = maximum(fb)

    if (test_bound.start > 1) || (test_bound.stop < var_len)
        iter_slice1 = bound_slice(iter_bound1)
        iter_slice2 = bound_slice(iter_bound2)

        if test_axis == 0
            min_slice = @view fb[test_bound.start, iter_slice1, iter_slice2]
            max_slice = @view fb[test_bound.stop, iter_slice1, iter_slice2]
        elseif test_axis == 1
            min_slice = @view fb[iter_slice1, test_bound.start, iter_slice2]
            max_slice = @view fb[iter_slice1, test_bound.stop, iter_slice2]
        elseif test_axis == 2
            min_slice = @view fb[iter_slice1, iter_slice2, test_bound.start]
            max_slice = @view fb[iter_slice1, iter_slice2, test_bound.stop]
        else
            error("Invalid test axis")
        end

        threshold = do_warn * big
        if (test_bound.start > 1) && any(x -> x > threshold, min_slice)
            @warn "Non-zero value of fb detected at min($var_name) boundary"
        end
        if (test_bound.stop < var_len) && any(x -> x > threshold, max_slice)
            @warn "Non-zero value of fb detected at max($var_name) boundary"
        end
    end

    return nothing
end

function _interp_fvrxx_weight_matrix(
    mesh_a::KineticMesh,
    mesh_b::KineticMesh,
    vdiff_a::VSpaceDifferentials,
    vdiff_b::VSpaceDifferentials,
    v_scale::Float64,
)::Matrix{Float64}
    nvr_a = length(mesh_a.vr)
    nvx_a = length(mesh_a.vx)
    nvr_b = length(mesh_b.vr)
    nvx_b = length(mesh_b.vx)

    b_vr_l = Vector{Float64}(undef, nvr_b)
    b_vr_r = Vector{Float64}(undef, nvr_b)
    b_vx_l = Vector{Float64}(undef, nvx_b)
    b_vx_r = Vector{Float64}(undef, nvx_b)

    @inbounds for i in 1:nvr_b
        b_vr_l[i] = v_scale * vdiff_b.vr_left_bound[i]
        b_vr_r[i] = v_scale * vdiff_b.vr_right_bound[i]
    end
    @inbounds for j in 1:nvx_b
        b_vx_l[j] = v_scale * vdiff_b.vx_left_bound[j]
        b_vx_r[j] = v_scale * vdiff_b.vx_right_bound[j]
    end

    weight = zeros(Float64, nvr_b * nvx_b, nvr_a * nvx_a)

    Threads.@threads for col in 1:(nvr_a * nvx_a)
        ia_vr = ((col - 1) % nvr_a) + 1
        ia_vx = ((col - 1) ÷ nvr_a) + 1

        a_vr_l = vdiff_a.vr_left_bound[ia_vr]
        a_vr_r = vdiff_a.vr_right_bound[ia_vr]
        a_vx_l = vdiff_a.vx_left_bound[ia_vx]
        a_vx_r = vdiff_a.vx_right_bound[ia_vx]

        @inbounds for ib_vx in 1:nvx_b
            vx_min = max(a_vx_l, b_vx_l[ib_vx])
            vx_max = min(a_vx_r, b_vx_r[ib_vx])
            vx_width = vx_max - vx_min
            vx_width <= 0.0 && continue

            denom_vx = vdiff_b.dvx[ib_vx]
            for ib_vr in 1:nvr_b
                vr_min = max(a_vr_l, b_vr_l[ib_vr])
                vr_max = min(a_vr_r, b_vr_r[ib_vr])
                vr_max <= vr_min && continue

                row = (ib_vx - 1) * nvr_b + ib_vr
                weight[row, col] =
                    2.0π * (vr_max^2 - vr_min^2) * vx_width /
                    (vdiff_b.dvr_vol[ib_vr] * denom_vx)
            end
        end
    end

    return weight
end

function _interp_fvrxx_moments(
    fa::AbstractArray{<:Real,3},
    mesh_a::KineticMesh,
    vdiff_a::VSpaceDifferentials,
)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    nx = length(mesh_a.x)
    density = zeros(Float64, nx)
    vx_moment = zeros(Float64, nx)
    energy_moment = zeros(Float64, nx)
    sqrt_Tnorm = sqrt(mesh_a.Tnorm)
    Tnorm = mesh_a.Tnorm

    Threads.@threads for k in 1:nx
        density_k = 0.0
        vx_num = 0.0
        energy_num = 0.0

        @inbounds for j in eachindex(mesh_a.vx)
            vxj = mesh_a.vx[j]
            dvxj = vdiff_a.dvx[j]
            for i in eachindex(mesh_a.vr)
                fij = Float64(fa[i, j, k])
                w = vdiff_a.dvr_vol[i] * dvxj * fij
                density_k += w
                vx_num += vxj * w
                energy_num += vdiff_a.vmag_squared[i, j] * w
            end
        end

        density[k] = density_k
        if density_k > 0.0
            vx_moment[k] = sqrt_Tnorm * vx_num / density_k
            energy_moment[k] = Tnorm * energy_num / density_k
        end
    end

    return density, vx_moment, energy_moment
end

function _interp_fvrxx_fill_on_mesh_b!(
    fb::Array{Float64,3},
    target_vx::Vector{Float64},
    target_energy::Vector{Float64},
    fb_on_xa::Matrix{Float64},
    vx_moment_on_xa::Vector{Float64},
    energy_moment_on_xa::Vector{Float64},
    mesh_a::KineticMesh,
    mesh_b::KineticMesh,
    x_bound::Bound,
)::Nothing
    nvr_b = length(mesh_b.vr)
    nvx_b = length(mesh_b.vx)
    nx_a = length(mesh_a.x)

    Threads.@threads for k in x_bound.start:x_bound.stop
        position0 = max(_locate_python_style(mesh_a.x, mesh_b.x[k]), 0)
        kr0 = min(position0 + 1, nx_a - 1)
        kl0 = min(position0, kr0 - 1)
        kl = kl0 + 1
        kr = kr0 + 1

        interp_fraction = (mesh_b.x[k] - mesh_a.x[kl]) / (mesh_a.x[kr] - mesh_a.x[kl])

        @inbounds for j in 1:nvx_b
            for i in 1:nvr_b
                row = (j - 1) * nvr_b + i
                left_val = fb_on_xa[row, kl]
                right_val = fb_on_xa[row, kr]
                fb[i, j, k] = left_val + interp_fraction * (right_val - left_val)
            end
        end

        target_vx[k] =
            vx_moment_on_xa[kl] +
            interp_fraction * (vx_moment_on_xa[kr] - vx_moment_on_xa[kl])
        target_energy[k] =
            energy_moment_on_xa[kl] +
            interp_fraction * (energy_moment_on_xa[kr] - energy_moment_on_xa[kl])
    end

    return nothing
end

function _interp_fvrxx_compensate!(
    fb::Array{Float64,3},
    target_vx::Vector{Float64},
    target_energy::Vector{Float64},
    mesh_b::KineticMesh,
    vdiff_b::VSpaceDifferentials,
    x_bound::Bound,
)::Nothing
    nvr_b = length(mesh_b.vr)
    nvx_b = length(mesh_b.vx)
    sqrt_Tnorm_b = sqrt(mesh_b.Tnorm)
    scratches = [MaxwellianScratch(nvr_b, nvx_b) for _ in 1:Threads.nthreads()]

    Threads.@threads for k in x_bound.start:x_bound.stop
        nb = 0.0
        @inbounds for j in eachindex(mesh_b.vx)
            dvxj = vdiff_b.dvx[j]
            for i in eachindex(mesh_b.vr)
                nb += vdiff_b.dvr_vol[i] * fb[i, j, k] * dvxj
            end
        end

        nb <= 0.0 && continue
        scratch = scratches[Threads.threadid()]
        out = scratch.out
        slice = @view fb[:, :, k]

        while true
            s = compensate_distribution!(
                out,
                scratch,
                slice,
                vdiff_b,
                mesh_b.vr,
                mesh_b.vx,
                sqrt_Tnorm_b,
                target_vx[k],
                target_energy[k];
                nb=nb,
                assume_pos=false,
            )
            slice .= out
            s >= 1.0 && break
        end
    end

    return nothing
end

function _interp_fvrxx_rescale!(
    fb::Array{Float64,3},
    fa::AbstractArray{<:Real,3},
    mesh_a::KineticMesh,
    mesh_b::KineticMesh,
    vdiff_a::VSpaceDifferentials,
    vdiff_b::VSpaceDifferentials,
    x_bound::Bound,
)::Tuple{Vector{Float64},Vector{Float64}}
    tot_a = zeros(Float64, length(mesh_a.x))
    tot_b = zeros(Float64, length(mesh_b.x))

    Threads.@threads for k in eachindex(mesh_a.x)
        sum_a = 0.0
        @inbounds for j in eachindex(mesh_a.vx)
            dvxj = vdiff_a.dvx[j]
            for i in eachindex(mesh_a.vr)
                sum_a += vdiff_a.dvr_vol[i] * dvxj * Float64(fa[i, j, k])
            end
        end
        tot_a[k] = sum_a
    end

    xb_range = x_bound.start:x_bound.stop
    tot_b[xb_range] .= interp_1d(
        mesh_a.x,
        tot_a,
        mesh_b.x[xb_range];
        fill_value="extrapolate",
    )

    min_tot = Inf
    @inbounds for val in fb
        if val > 0.0 && val < min_tot
            min_tot = val
        end
    end

    if isfinite(min_tot)
        Threads.@threads for k in xb_range
            tot = 0.0
            @inbounds for j in eachindex(mesh_b.vx)
                dvxj = vdiff_b.dvx[j]
                for i in eachindex(mesh_b.vr)
                    tot += vdiff_b.dvr_vol[i] * dvxj * fb[i, j, k]
                end
            end

            if tot > min_tot
                factor = tot_b[k] / tot
                @inbounds for j in eachindex(mesh_b.vx), i in eachindex(mesh_b.vr)
                    fb[i, j, k] *= factor
                end
            end
        end
    end

    return tot_a, tot_b
end

function interp_fvrxx(
    fa::AbstractArray{<:Real,3},
    mesh_a::KineticMesh,
    mesh_b::KineticMesh;
    do_warn::Union{Nothing,Real}=nothing,
    debug::Bool=false,
    correct::Int=1,
)::Array{Float64,3}
    prompt = "INTERP_FVRXX => "
    nvr_b = length(mesh_b.vr)
    nvx_b = length(mesh_b.vx)

    v_scale = sqrt(mesh_b.Tnorm / mesh_a.Tnorm)

    if size(fa) != (length(mesh_a.vr), length(mesh_a.vx), length(mesh_a.x))
        error("Input array size does not match mesh_a dimensions")
    end

    scaled_vr_b = v_scale .* mesh_b.vr
    scaled_vx_b = v_scale .* mesh_b.vx
    vr_bound = _get_interpolation_bounds(mesh_a.vr, scaled_vr_b, a_name="mesh_a.vr", b_name="mesh_b.vr")
    vx_bound = _get_interpolation_bounds(mesh_a.vx, scaled_vx_b, a_name="mesh_a.vx", b_name="mesh_b.vx")
    x_bound = _get_interpolation_bounds(mesh_a.x, mesh_b.x, a_name="mesh_a.x", b_name="mesh_b.x")

    fb = zeros(Float64, nvr_b, nvx_b, length(mesh_b.x))
    vdiff_a = VSpaceDifferentials(mesh_a.vr, mesh_a.vx)
    vdiff_b = VSpaceDifferentials(mesh_b.vr, mesh_b.vx)

    debug && println(prompt * "computing new weight")
    weight = _interp_fvrxx_weight_matrix(mesh_a, mesh_b, vdiff_a, vdiff_b, v_scale)

    if correct == 1
        fa_reshaped = reshape(fa, :, size(fa, 3))
        fb_on_xa = weight * fa_reshaped
        _, vx_moment_on_xa, energy_moment_on_xa = _interp_fvrxx_moments(fa, mesh_a, vdiff_a)

        target_vx = zeros(Float64, length(mesh_b.x))
        target_energy = zeros(Float64, length(mesh_b.x))
        _interp_fvrxx_fill_on_mesh_b!(
            fb,
            target_vx,
            target_energy,
            fb_on_xa,
            vx_moment_on_xa,
            energy_moment_on_xa,
            mesh_a,
            mesh_b,
            x_bound,
        )
        _interp_fvrxx_compensate!(fb, target_vx, target_energy, mesh_b, vdiff_b, x_bound)
    end

    if do_warn !== nothing
        _test_bounds(fb, vr_bound, nvr_b, 0, vx_bound, x_bound, do_warn, "vr")
        _test_bounds(fb, vx_bound, nvx_b, 1, vr_bound, x_bound, do_warn, "vx")
        _test_bounds(fb, x_bound, length(mesh_b.x), 2, vr_bound, vx_bound, do_warn, "x")
    end

    _interp_fvrxx_rescale!(fb, fa, mesh_a, mesh_b, vdiff_a, vdiff_b, x_bound)
    return fb
end
