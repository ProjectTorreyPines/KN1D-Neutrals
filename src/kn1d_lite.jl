struct KN1DLiteResults
    xH::Vector{Float64}
    vr::Vector{Float64}
    vx::Vector{Float64}
    Tnorm::Float64
    fH::Array{Float64,3}
    nH::Vector{Float64}
    GammaxH::Vector{Float64}
    VxH::Vector{Float64}
    TH::Vector{Float64}
    qxH_total::Vector{Float64}
    Sion::Vector{Float64}
    fHBC::Matrix{Float64}
    GammaxHBC::Float64
end

mutable struct KN1DLiteProblem
    kh::KineticH
    fHBC::Matrix{Float64}
    GammaxHBC::Float64
    fH2A::Array{Float64,3}
    fSHA::Array{Float64,3}
    fH_init::Array{Float64,3}
    nHPA::Vector{Float64}
    THPA::Vector{Float64}
end

const DEFAULT_JH_CACHE = Ref{Union{Nothing,JohnsonHinnov}}(nothing)

function default_johnson_hinnov()::JohnsonHinnov
    jh = DEFAULT_JH_CACHE[]
    if jh === nothing
        jh = JohnsonHinnov()
        DEFAULT_JH_CACHE[] = jh
    end
    return jh
end

function _as_float_vector(x, name::AbstractString)::Vector{Float64}
    v = Float64.(x)
    all(isfinite, v) || throw(ArgumentError("$name must contain only finite values"))
    return Vector{Float64}(v)
end

function _validate_profile_lengths(
    x::Vector{Float64},
    Ti::Vector{Float64},
    Te::Vector{Float64},
    n::Vector{Float64},
    vxi::Vector{Float64},
)::Nothing
    nx = length(x)
    nx >= 2 || throw(ArgumentError("x must contain at least two points"))
    length(Ti) == nx || throw(DimensionMismatch("Ti length must match x"))
    length(Te) == nx || throw(DimensionMismatch("Te length must match x"))
    length(n) == nx || throw(DimensionMismatch("n length must match x"))
    length(vxi) == nx || throw(DimensionMismatch("vxi length must match x"))
    all(diff(x) .> 0.0) || throw(ArgumentError("x must be strictly increasing"))
    return nothing
end

function _component_speeds(
    mu::Int,
    energies_eV,
    velocities_ms,
    fractions,
)::Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    if velocities_ms !== nothing && energies_eV !== nothing
        @warn "KN1D_lite => both velocities_ms and energies_eV supplied; velocities_ms takes precedence."
    end

    component_vs = if velocities_ms !== nothing
        _as_float_vector(velocities_ms, "velocities_ms")
    else
        energies = _as_float_vector(energies_eV === nothing ? [3.0] : energies_eV, "energies_eV")
        @inbounds for val in energies
            val >= 0.0 || throw(ArgumentError("energies_eV must be non-negative"))
        end
        sqrt.((2.0 * Q / (mu * H_MASS)) .* energies)
    end

    isempty(component_vs) && throw(ArgumentError("at least one incident component is required"))
    @inbounds for val in component_vs
        val >= 0.0 || throw(ArgumentError("velocities_ms must be non-negative"))
    end

    frac = fractions === nothing ? fill(1.0 / length(component_vs), length(component_vs)) :
           _as_float_vector(fractions, "fractions")
    length(frac) == length(component_vs) ||
        throw(DimensionMismatch("fractions and energies/velocities must have the same length"))
    isapprox(sum(frac), 1.0; atol=1.0e-12, rtol=1.0e-12) ||
        throw(ArgumentError("fractions must sum to 1.0; got $(sum(frac))"))

    E0 = if velocities_ms !== nothing
        @. 0.5 * mu * H_MASS * component_vs^2 / Q
    else
        _as_float_vector(energies_eV === nothing ? [3.0] : energies_eV, "energies_eV")
    end

    return component_vs, frac, E0
end

function _lite_result(problem::KN1DLiteProblem, kh_results::KHResults)::KN1DLiteResults
    kh = problem.kh
    return KN1DLiteResults(
        kh.mesh.x,
        kh.mesh.vr,
        kh.mesh.vx,
        kh.mesh.Tnorm,
        kh_results.fH,
        kh_results.nH,
        kh_results.GammaxH,
        kh_results.VxH,
        kh_results.TH,
        kh_results.qxH_total,
        kh_results.Sion,
        problem.fHBC,
        problem.GammaxHBC,
    )
end

function prepare_kn1d_lite(
    x::AbstractVector{<:Real},
    mu::Real,
    Ti::AbstractVector{<:Real},
    Te::AbstractVector{<:Real},
    n::AbstractVector{<:Real},
    vxi::AbstractVector{<:Real},
    incident_n0::Real;
    energies_eV=nothing,
    velocities_ms=nothing,
    fractions=nothing,
    fH_BC=nothing,
    truncate::Real=1.0e-3,
    max_gen::Integer=50,
    compute_errors::Bool=false,
    debrief::Bool=false,
    debug::Bool=false,
    config_path::AbstractString="./config.json",
    jh::Union{Nothing,JohnsonHinnov}=nothing,
)::KN1DLiteProblem
    prompt = "KN1D_lite => "

    x_f = _as_float_vector(x, "x")
    Ti_f = _as_float_vector(Ti, "Ti")
    Te_f = _as_float_vector(Te, "Te")
    n_f = _as_float_vector(n, "n")
    vxi_f = _as_float_vector(vxi, "vxi")
    _validate_profile_lengths(x_f, Ti_f, Te_f, n_f, vxi_f)

    mu_i = Int(mu)
    Float64(mu) == Float64(mu_i) || throw(ArgumentError("mu must be 1 for hydrogen or 2 for deuterium"))
    mu_i in (1, 2) || throw(ArgumentError("mu must be 1 for hydrogen or 2 for deuterium"))

    incident = Float64(incident_n0)
    incident >= 0.0 || throw(ArgumentError("incident_n0 must be non-negative"))

    advanced_mode = fH_BC !== nothing
    if advanced_mode && (energies_eV !== nothing || velocities_ms !== nothing || fractions !== nothing)
        @warn prompt * "fH_BC supplied; energies_eV/velocities_ms/fractions are ignored."
    end

    component_vs = Float64[]
    component_fractions = Float64[]
    E0 = advanced_mode ? [0.0] : Float64[]
    if !advanced_mode
        component_vs, component_fractions, E0 = _component_speeds(mu_i, energies_eV, velocities_ms, fractions)
    end

    cfg = get_config(String(config_path))
    jh_obj = jh !== nothing ? jh :
             cfg.kinetic_h.ion_rate == "jh" ? default_johnson_hinnov() :
             nothing

    kh_mesh = KineticMesh(
        "h",
        mu_i,
        x_f,
        Ti_f,
        Te_f,
        n_f,
        zeros(Float64, length(x_f));
        jh=jh_obj,
        E0=E0,
        config_path=String(config_path),
    )

    vth = sqrt(2.0 * Q * kh_mesh.Tnorm / (mu_i * H_MASS))
    kh_differentials = VSpaceDifferentials(kh_mesh.vr, kh_mesh.vx)
    fHBC = zeros(Float64, length(kh_mesh.vr), length(kh_mesh.vx))
    GammaxHBC = 0.0

    if !advanced_mode
        @inbounds for m in eachindex(component_vs, component_fractions)
            v_ms = component_vs[m]
            v_norm = v_ms / vth
            ix = argmin(abs.(kh_mesh.vx .- v_norm))
            fHBC[1, ix] += (component_fractions[m] * incident) /
                            (kh_differentials.dvr_vol[1] * kh_differentials.dvx[ix])
            GammaxHBC += component_fractions[m] * incident * v_ms
        end
    else
        f_in = Matrix{Float64}(Float64.(fH_BC))
        size(f_in) == size(fHBC) ||
            throw(DimensionMismatch("fH_BC must have size ($(size(fHBC, 1)), $(size(fHBC, 2))) for the generated velocity mesh"))

        @inbounds for j in eachindex(kh_mesh.vx)
            if kh_mesh.vx[j] < 0.0
                for i in eachindex(kh_mesh.vr)
                    f_in[i, j] = 0.0
                end
            end
        end

        current_n = 0.0
        @inbounds for j in eachindex(kh_mesh.vx)
            dvxj = kh_differentials.dvx[j]
            for i in eachindex(kh_mesh.vr)
                current_n += kh_differentials.dvr_vol[i] * f_in[i, j] * dvxj
            end
        end
        current_n > 0.0 ||
            throw(ArgumentError(prompt * "fH_BC integrates to zero or negative density after zeroing negative vx."))

        scale = incident / current_n
        @inbounds for j in axes(fHBC, 2), i in axes(fHBC, 1)
            fHBC[i, j] = f_in[i, j] * scale
        end

        @inbounds for j in eachindex(kh_mesh.vx)
            pos_vx_flux = max(kh_mesh.vx[j], 0.0) * kh_differentials.dvx[j]
            for i in eachindex(kh_mesh.vr)
                GammaxHBC += kh_differentials.dvr_vol[i] * fHBC[i, j] * pos_vx_flux
            end
        end
        GammaxHBC *= vth
    end

    vxiA = interp_1d(x_f, vxi_f, kh_mesh.x; fill_value="extrapolate")
    kh = KineticH(
        kh_mesh,
        mu_i,
        vxiA,
        fHBC,
        GammaxHBC;
        jh=jh_obj,
        ni_correct=true,
        truncate=Float64(truncate),
        max_gen=Int(max_gen),
        compute_errors=compute_errors,
        debrief=debrief ? 1 : 0,
        debug=debug ? 1 : 0,
        config_path=String(config_path),
        initialize_static=true,
    )

    return KN1DLiteProblem(
        kh,
        fHBC,
        GammaxHBC,
        zeros(Float64, kh.nvr, kh.nvx, kh.nx),
        zeros(Float64, kh.nvr, kh.nvx, kh.nx),
        zeros(Float64, kh.nvr, kh.nvx, kh.nx),
        zeros(Float64, kh.nx),
        ones(Float64, kh.nx),
    )
end

function _reset_lite_dynamic_state!(kh::KineticH)::Nothing
    kh.collisions = KHCollisions(kh.config.collisions)
    kh.input = KineticHInput()
    kh.h2_moments = KineticHH2Moments()
    if kh.internal.MH_H_sum === nothing
        kh.internal.MH_H_sum = zeros(Float64, kh.nvr, kh.nvx, kh.nx)
    else
        fill!(kh.internal.MH_H_sum, 0.0)
    end
    kh.internal.Delta_nHs = 1.0
    return nothing
end

function run_kn1d_lite(problem::KN1DLiteProblem)::KN1DLiteResults
    kh = problem.kh
    _reset_lite_dynamic_state!(kh)

    fill!(problem.fH2A, 0.0)
    fill!(problem.fSHA, 0.0)
    fill!(problem.fH_init, 0.0)
    fill!(problem.nHPA, 0.0)
    fill!(problem.THPA, 1.0)

    kh_results = run_procedure(
        kh;
        fH2=problem.fH2A,
        fSH=problem.fSHA,
        fH=problem.fH_init,
        nHP=problem.nHPA,
        THP=problem.THPA,
    )
    return _lite_result(problem, kh_results)
end

function kn1d_lite(args...; kwargs...)::KN1DLiteResults
    return run_kn1d_lite(prepare_kn1d_lite(args...; kwargs...))
end
