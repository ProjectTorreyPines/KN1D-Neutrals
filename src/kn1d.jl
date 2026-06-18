struct KN1DResults
    # Molecular Info
    xH2::Vector{Float64}
    nH2::Vector{Float64}
    GammaxH2::Vector{Float64}
    TH2::Vector{Float64}
    qxH2_total::Vector{Float64}
    nHP::Vector{Float64}
    THP::Vector{Float64}
    SH::Vector{Float64}
    SP::Vector{Float64}

    # Atomic Info
    xH::Vector{Float64}
    nH::Vector{Float64}
    GammaxH::Vector{Float64}
    TH::Vector{Float64}
    qxH_total::Vector{Float64}
    NetHSource::Vector{Float64}
    Sion::Vector{Float64}
    QH_total::Vector{Float64}
    SideWallH::Vector{Float64}
    Lyman::Vector{Float64}
    Balmer::Vector{Float64}

    # Combined
    GammaHLim::Float64
end

function trapz(x::AbstractVector, y::AbstractVector)
    s = zero(eltype(y))
    @inbounds for i in 1:length(x)-1
        s += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    end
    return s
end

function kn1d(
    x::Vector{Float64},
    xlimiter::Float64,
    xsep::Float64,
    GaugeH2::Float64,
    mu::Integer,
    Ti::Vector{Float64},
    Te::Vector{Float64},
    n::Vector{Float64},
    vxi::Vector{Float64},
    LC::Vector{Float64},
    PipeDia::Vector{Float64};
    truncate::Float64 = 1.0e-3,
    max_gen::Int = 100,
    compute_errors::Bool = false,
    debrief::Bool = false,
    Hdebug::Bool = false,
    Hdebrief::Bool = false,
    H2debug::Bool = false,
    H2debrief::Bool = false,
    interp_debug::Bool = false,
    File::Union{Nothing,String} = nothing,
    config_path::String = "./config.json",
)
    prompt = "KN1D => "

    # -------------------------------------------------------------------------
    # Validate config options
    # -------------------------------------------------------------------------

    valid_ion_rates = ["collrad", "jh", "janev", "adas"]

    cfg = get_config(config_path)

    ion_rate_option = cfg.kinetic_h.ion_rate
    grid_fctr_h2 = cfg.kinetic_h2.grid_fctr
    grid_fctr_h  = cfg.kinetic_h.grid_fctr
    extra_bins_h2 = cfg.kinetic_h2.extra_energy_bins_eV
    extra_bins_h = cfg.kinetic_h.extra_energy_bins_eV

    if !(ion_rate_option in valid_ion_rates)
        error(prompt * "Invalid Ionization Rate Option used: '" *
              string(ion_rate_option) * "', check config.json")
    end

    # -------------------------------------------------------------------------
    # Generate meshes
    # -------------------------------------------------------------------------

    Eneut = unique(sort(vcat([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], extra_bins_h2)))

    fctr_h2 = grid_fctr_h2
    if GaugeH2 > 15.0
        fctr_h2 *= 15.0 / GaugeH2
    end

    kh2_mesh = KineticMesh(
        "h2",
        mu,
        x,
        Ti,
        Te,
        n,
        PipeDia;
        E0 = Eneut,
        fctr = fctr_h2,
        config_path = config_path,
    )

    E0_h = !isempty(extra_bins_h) ? unique(sort(extra_bins_h)) : [0.0]

    fctr_h = grid_fctr_h
    if GaugeH2 > 30.0
        fctr_h *= 30.0 / GaugeH2
    end

    # Johnson-Hinnov object, replacing IDL JH_Coef common block
    jh = default_johnson_hinnov()

    kh_mesh = KineticMesh(
        "h",
        mu,
        x,
        Ti,
        Te,
        n,
        PipeDia;
        jh = jh,
        E0 = E0_h,
        fctr = fctr_h,
        config_path = config_path,
    )

    # -------------------------------------------------------------------------
    # Initialize variables
    # -------------------------------------------------------------------------

    fH = zeros(Float64, length(kh_mesh.vr), length(kh_mesh.vx), length(kh_mesh.x))
    fH2 = zeros(Float64, length(kh2_mesh.vr), length(kh2_mesh.vx), length(kh2_mesh.x))

    nH2 = zeros(Float64, length(kh2_mesh.x))
    nHP = zeros(Float64, length(kh2_mesh.x))
    THP = zeros(Float64, length(kh2_mesh.x))

    # Directed random velocity of diatomic molecule
    v0_bar = sqrt((8.0 * TWALL * Q) / (π * 2.0 * mu * H_MASS))

    # Set up molecular flux BC from inputted neutral pressure
    ipM = findall(>(0.0), kh2_mesh.vx)

    fh2BC = zeros(Float64, length(kh2_mesh.vr), length(kh2_mesh.vx))

    # Convert pressure [mtorr] to molecular density and flux
    DensM = 3.537e19 * GaugeH2
    GammaxH2BC = 0.25 * DensM * v0_bar

    Tmaxwell = [TWALL]
    vx_shift = [0.0]

    Maxwell = create_shifted_maxwellian(
        kh2_mesh.vr,
        kh2_mesh.vx,
        Tmaxwell,
        vx_shift,
        mu,
        2,
        kh2_mesh.Tnorm,
    )

    @views fh2BC[:, ipM] .= Maxwell[:, ipM, 1]

    # -------------------------------------------------------------------------
    # Compute NuLoss = Cs / LC
    # -------------------------------------------------------------------------

    Cs_LC = zeros(Float64, length(LC))

    @inbounds for ii in eachindex(LC)
        if LC[ii] > 0.0
            Cs_LC[ii] = sqrt(Q * (Ti[ii] + Te[ii]) / (mu * H_MASS)) / LC[ii]
        end
    end

    NuLoss = interp_1d(x, Cs_LC, kh2_mesh.x)

    # -------------------------------------------------------------------------
    # Compute first guess SpH2
    #
    # Integral{SpH2} dx = (2/3) GammaxH2BC
    # SpH2 = beta * n * Cs / LC
    # -------------------------------------------------------------------------

    SpH2_hat = interp_1d(x, n .* Cs_LC, kh2_mesh.x; fill_value = "extrapolate")

    SpH2_hat ./= trapz(kh2_mesh.x, SpH2_hat)

    beta = (2.0 / 3.0) * GammaxH2BC

    SpH2 = beta .* SpH2_hat
    SH2 = copy(SpH2)

    # -------------------------------------------------------------------------
    # Interpolate vxi for molecular and atomic meshes
    # -------------------------------------------------------------------------

    vxiM = interp_1d(x, vxi, kh2_mesh.x; fill_value = "extrapolate")
    vxiA = interp_1d(x, vxi, kh_mesh.x; fill_value = "extrapolate")

    # -------------------------------------------------------------------------
    # Compute velocity-space differentials
    # -------------------------------------------------------------------------

    vthM = sqrt(2.0 * Q * kh2_mesh.Tnorm / (mu * H_MASS))
    kh2_differentials = VSpaceDifferentials(kh2_mesh.vr, kh2_mesh.vx)

    vthA = sqrt(2.0 * Q * kh_mesh.Tnorm / (mu * H_MASS))
    kh_differentials = VSpaceDifferentials(kh_mesh.vr, kh_mesh.vx)

    # -------------------------------------------------------------------------
    # Test wall Maxwellian consistency
    # -------------------------------------------------------------------------

    nbarHMax = sum(kh2_differentials.dvr_vol .* (fh2BC * kh2_differentials.dvx))

    vbarM =
        2.0 * vthM *
        sum(kh2_differentials.dvr_vol .* (fh2BC * (kh2_mesh.vx .* kh2_differentials.dvx))) /
        nbarHMax

    vbarM_error = abs(vbarM - v0_bar) / max(vbarM, v0_bar)

    vr2vx2_ran2 = zeros(Float64, length(kh2_mesh.vr), length(kh2_mesh.vx))

    @views mwell = Maxwell[:, :, 1]

    nbarMax = sum(kh2_differentials.dvr_vol .* (mwell * kh2_differentials.dvx))

    UxMax =
        vthM *
        sum(kh2_differentials.dvr_vol .* (mwell * (kh2_mesh.vx .* kh2_differentials.dvx))) /
        nbarMax

    @inbounds for i in eachindex(kh2_mesh.vr)
        vr2vx2_ran2[i, :] .= kh2_mesh.vr[i]^2 .+ (kh2_mesh.vx .- UxMax / vthM).^2
    end

    TMax =
        2.0 * mu * H_MASS * vthM^2 *
        sum(kh2_differentials.dvr_vol .* ((vr2vx2_ran2 .* mwell) * kh2_differentials.dvx)) /
        (3.0 * Q * nbarMax)

    UxHMax =
        vthM *
        sum(kh2_differentials.dvr_vol .* (fh2BC * (kh2_mesh.vx .* kh2_differentials.dvx))) /
        nbarHMax

    @inbounds for i in eachindex(kh2_mesh.vr)
        vr2vx2_ran2[i, :] .= kh2_mesh.vr[i]^2 .+ (kh2_mesh.vx .- UxHMax / vthM).^2
    end

    THMax =
        2.0 * mu * H_MASS * vthM^2 *
        sum(kh2_differentials.dvr_vol .* ((vr2vx2_ran2 .* fh2BC) * kh2_differentials.dvx)) /
        (3.0 * Q * nbarHMax)

    if compute_errors && debrief
        println(prompt * "VbarM_error: " * sval(vbarM_error))
        println(prompt * "TWall Maxwellian: " * sval(TMax))
        println(prompt * "TWall Half Maxwellian: " * sval(THMax))
    end

    # -------------------------------------------------------------------------
    # Setup procedure classes
    # -------------------------------------------------------------------------

    GammaxHBC = 0.0
    fHBC = zeros(Float64, length(kh_mesh.vr), length(kh_mesh.vx))

    kinetic_h = KineticH(
        kh_mesh,
        mu,
        vxiA,
        fHBC,
        GammaxHBC;
        jh = jh,
        ni_correct = true,
        truncate = truncate,
        max_gen = max_gen,
        compute_errors = compute_errors,
        debrief = Hdebrief,
        debug = Hdebug,
        config_path = config_path,
    )

    kinetic_h2 = KineticH2(
        kh2_mesh,
        mu,
        vxiM,
        fh2BC,
        GammaxH2BC,
        NuLoss,
        SH2;
        compute_h_source = true,
        ni_correct = true,
        truncate = truncate,
        max_gen = max_gen,
        compute_errors = compute_errors,
        debrief = H2debrief,
        debug = H2debug,
        config_path = config_path,
    )

    # -------------------------------------------------------------------------
    # Begin iteration
    # -------------------------------------------------------------------------

    println(prompt * "Satisfaction condition: ", truncate)

    iter = 0

    EH_hist = Float64[0.0]
    SI_hist = Float64[0.0]

    kh2_results = nothing
    kh_results = nothing

    while true
        iter += 1

        if debrief
            println(prompt * "fH/fH2 Iteration: " * sval(iter))
        end

        nH2_saved = copy(nH2)

        # Interpolate fH onto H2 mesh: fH -> fHM
        do_warn = 5e-3
        fHM = interp_fvrxx(fH, kh_mesh, kh2_mesh; do_warn = do_warn, debug = interp_debug)

        # ---------------------------------------------------------------------
        # Run kinetic_h2
        # ---------------------------------------------------------------------

        kh2_results = run_procedure(kinetic_h2; fH = fHM, SH2 = SH2, fH2 = fH2, nHP = nHP, THP = THP)

        fH2 = kh2_results.fH2
        nHP = kh2_results.nHP
        THP = kh2_results.THP
        nH2 = kh2_results.nH2

        # Interpolate H2 data onto H mesh
        do_warn = 5.0e-3

        fH2A = interp_fvrxx(fH2, kh2_mesh, kh_mesh; do_warn = do_warn, debug = interp_debug)

        fSHA = interp_fvrxx(
            kh2_results.fSH,
            kh2_mesh,
            kh_mesh;
            do_warn = do_warn,
            debug = interp_debug,
        )

        nHPA = interp_1d(kh2_mesh.x, nHP, kh_mesh.x; fill_value = 0.0)
        THPA = interp_1d(kh2_mesh.x, THP, kh_mesh.x; fill_value = 0.0)

        # ---------------------------------------------------------------------
        # Run kinetic_h
        # ---------------------------------------------------------------------

        kh_results = run_procedure(kinetic_h; fH2 = fH2A, fSH = fSHA, fH = fH, nHP = nHPA, THP = THPA)

        fH = kh_results.fH

        # Interpolate SideWallH onto H2 mesh
        SideWallHM = interp_1d(kh_mesh.x, kh_results.SideWallH, kh2_mesh.x; fill_value = 0.0)

        # ---------------------------------------------------------------------
        # Adjust SpH2 to achieve net zero hydrogen atom/molecule flux from wall
        # ---------------------------------------------------------------------

        SI = trapz(kh2_mesh.x, SpH2)
        SwallI = trapz(kh2_mesh.x, 0.5 .* SideWallHM)

        GammaH2Wall_minus = kh2_results.AlbedoH2 * GammaxH2BC
        GammaHWall_minus = -kh_results.GammaxH[1]

        Epsilon = 2.0 * GammaH2Wall_minus / (SI + SwallI)

        alphaplus1RH0Dis =
            GammaHWall_minus /
            ((1.0 - 0.5 * Epsilon) * (SI + SwallI) + GammaxH2BC)

        EH = 2.0 * kh2_results.GammaxH2[1] - GammaHWall_minus

        dEHdSI = -Epsilon - alphaplus1RH0Dis * (1.0 - 0.5 * Epsilon)

        nEH = abs(EH) / maximum(abs.([2.0 * kh2_results.GammaxH2[1], GammaHWall_minus]))

        if debrief && compute_errors
            println(prompt, "Normalized Hydrogen Flux Error: ", sval(nEH))
        end

        Delta_SI = -EH / dEHdSI
        SI += Delta_SI

        SpH2 .= SI .* SpH2_hat

        push!(EH_hist, EH)
        push!(SI_hist, SI)

        # Set total H2 source
        SH2 = SpH2 .+ 0.5 .* SideWallHM

        if compute_errors
            _RxH_H2 = interp_1d(
                kh2_mesh.x,
                kinetic_h2.output.RxH_H2,
                kh_mesh.x;
                fill_value = 0.0,
            )

            DRx = _RxH_H2 .+ kinetic_h.output.RxH2_H

            nDRx =
                maximum(abs.(DRx)) /
                maximum(abs.(vcat(_RxH_H2, kinetic_h.output.RxH2_H)))

            if debrief
                println(prompt, "Normalized H2 <-> H Momentum Transfer Error: ", sval(nDRx))
            end
        end

        Delta_nH2 = abs.(kh2_results.nH2 .- nH2_saved)

        nDelta_nH2 = maximum(Delta_nH2 ./ maximum(kh2_results.nH2))

        if debrief
            println(prompt, "Maximum Normalized change in nH2: ", sval(nDelta_nH2))
        end

        if nDelta_nH2 <= truncate
            break
        end
    end

    # -------------------------------------------------------------------------
    # End iteration
    # -------------------------------------------------------------------------

    gamma_h2 = interp_1d(kh2_mesh.x, kh2_results.GammaxH2, kh_mesh.x)
    gam = 2.0 .* gamma_h2 .+ kh_results.GammaxH

    GammaHLim = interp_1d(kh_mesh.x, gam, xlimiter)

    # -------------------------------------------------------------------------
    # Compute Lyman and Balmer alpha
    # -------------------------------------------------------------------------

    Lyman = lyman_alpha(jh, kh_mesh.ne, kh_mesh.Te, kh_results.nH; no_null = true)
    Balmer = balmer_alpha(jh, kh_mesh.ne, kh_mesh.Te, kh_results.nH; no_null = true)

    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------

    out_dir = isnothing(File) ? joinpath("Results", "output") : File
    mkpath(out_dir)

    println(prompt, "Saving files to ", out_dir)

    # KN1D_input
    npzwrite(
        joinpath(out_dir, "KN1D_input.npz"),
        Dict(
            "x" => x,
            "xlimiter" => xlimiter,
            "xsep" => xsep,
            "GaugeH2" => GaugeH2,
            "mu" => mu,
            "Ti" => Ti,
            "Te" => Te,
            "n" => n,
            "vxi" => vxi,
            "LC" => LC,
            "PipeDia" => PipeDia,
            "truncate" => truncate,

            "xH2" => kh2_mesh.x,
            "TiM" => kh2_mesh.Ti,
            "TeM" => kh2_mesh.Te,
            "nM" => kh2_mesh.ne,
            "PipeDiaM" => kh2_mesh.PipeDia,
            "vxM" => kh2_mesh.vx,
            "vrM" => kh2_mesh.vr,
            "TnormM" => kh2_mesh.Tnorm,

            "xH" => kh_mesh.x,
            "TiA" => kh_mesh.Ti,
            "TeA" => kh_mesh.Te,
            "nA" => kh_mesh.ne,
            "PipeDiaA" => kh_mesh.PipeDia,
            "vxA" => kh_mesh.vx,
            "vrA" => kh_mesh.vr,
            "TnormA" => kh_mesh.Tnorm,
        ),
    )

    # KN1D_mesh
    npzwrite(
        joinpath(out_dir, "KN1D_mesh.npz"),
        Dict(
            "x_s" => x,
            "GaugeH2_s" => GaugeH2,
            "mu_s" => mu,
            "Ti_s" => Ti,
            "Te_s" => Te,
            "n_s" => n,
            "vxi_s" => vxi,
            "LC_s" => LC,
            "PipeDia_s" => PipeDia,

            "xH2_s" => kh2_mesh.x,
            "vxM_s" => kh2_mesh.vx,
            "vrM_s" => kh2_mesh.vr,
            "TnormM_s" => kh2_mesh.Tnorm,

            "xH_s" => kh_mesh.x,
            "vxA_s" => kh_mesh.vx,
            "vrA_s" => kh_mesh.vr,
            "TnormA_s" => kh_mesh.Tnorm,
        ),
    )

    # KN1D_H2
    npzwrite(
        joinpath(out_dir, "KN1D_H2.npz"),
        Dict(
            "xH2" => kh2_mesh.x,
            "fH2" => kh2_results.fH2,
            "nH2" => kh2_results.nH2,
            "GammaxH2" => kh2_results.GammaxH2,
            "VxH2" => kh2_results.VxH2,
            "pH2" => kh2_results.pH2,
            "TH2" => kh2_results.TH2,
            "qxH2" => kh2_results.qxH2,
            "qxH2_total" => kh2_results.qxH2_total,
            "Sloss" => kh2_results.Sloss,
            "QH2" => kh2_results.QH2,
            "RxH2" => kh2_results.RxH2,
            "QH2_total" => kh2_results.QH2_total,
            "AlbedoH2" => kh2_results.AlbedoH2,
            "nHP" => kh2_results.nHP,
            "THP" => kh2_results.THP,
            "fSH" => kh2_results.fSH,
            "SH" => kh2_results.SH,
            "SP" => kh2_results.SP,
            "SHP" => kh2_results.SHP,
            "NuE" => kh2_results.NuE,
            "NuDis" => kh2_results.NuDis,

            "piH2_xx" => kinetic_h2.output.piH2_xx,
            "piH2_yy" => kinetic_h2.output.piH2_yy,
            "piH2_zz" => kinetic_h2.output.piH2_zz,
            "RxH2CX" => kinetic_h2.output.RxH2CX,
            "RxH_H2" => kinetic_h2.output.RxH_H2,
            "RxP_H2" => kinetic_h2.output.RxP_H2,
            "RxW_H2" => kinetic_h2.output.RxW_H2,
            "EH2CX" => kinetic_h2.output.EH2CX,
            "EH_H2" => kinetic_h2.output.EH_H2,
            "EP_H2" => kinetic_h2.output.EP_H2,
            "EW_H2" => kinetic_h2.output.EW_H2,
            "Epara_PerpH2_H2" => kinetic_h2.output.Epara_PerpH2_H2,

            "GammaxH2_plus" => kh2_results.GammaxH2[1],
            "GammaxH2_minus" => kh2_results.GammaxH2[end],
        ),
    )

    # KN1D_H
    npzwrite(
        joinpath(out_dir, "KN1D_H.npz"),
        Dict(
            "xH" => kh_mesh.x,
            "fH" => kh_results.fH,
            "nH" => kh_results.nH,
            "GammaxH" => kh_results.GammaxH,
            "VxH" => kh_results.VxH,
            "pH" => kh_results.pH,
            "TH" => kh_results.TH,
            "qxH" => kh_results.qxH,
            "qxH_total" => kh_results.qxH_total,
            "NetHSource" => kh_results.NetHSource,
            "Sion" => kh_results.Sion,
            "SideWallH" => kh_results.SideWallH,
            "QH" => kh_results.QH,
            "RxH" => kh_results.RxH,
            "QH_total" => kh_results.QH_total,
            "AlbedoH" => kh_results.AlbedoH,
            "GammaHLim" => GammaHLim,

            "piH_xx" => kinetic_h.output.piH_xx,
            "piH_yy" => kinetic_h.output.piH_yy,
            "piH_zz" => kinetic_h.output.piH_zz,
            "RxHCX" => kinetic_h.output.RxHCX,
            "RxH2_H" => kinetic_h.output.RxH2_H,
            "RxP_H" => kinetic_h.output.RxP_H,
            "RxW_H" => kinetic_h.output.RxW_H,
            "EHCX" => kinetic_h.output.EHCX,
            "EH2_H" => kinetic_h.output.EH2_H,
            "EP_H" => kinetic_h.output.EP_H,
            "EW_H" => kinetic_h.output.EW_H,
            "Epara_PerpH_H" => kinetic_h.output.Epara_PerpH_H,
            "SourceH" => kinetic_h.output.SourceH,
            "SRecomb" => kinetic_h.output.SRecomb,
            "EH_hist" => EH_hist,
            "SI_hist" => SI_hist,
            "Lyman" => Lyman,
            "Balmer" => Balmer,
        ),
    )

    # Config snapshot
    config_snapshot = get_json(config_path)

    open(joinpath(out_dir, "config.json"), "w") do io
        JSON.print(io, config_snapshot, 4)
    end

    # -------------------------------------------------------------------------
    # Return result container
    # -------------------------------------------------------------------------

    results = KN1DResults(
        kh2_mesh.x,
        kh2_results.nH2,
        kh2_results.GammaxH2,
        kh2_results.TH2,
        kh2_results.qxH2_total,
        kh2_results.nHP,
        kh2_results.THP,
        kh2_results.SH,
        kh2_results.SP,

        kh_mesh.x,
        kh_results.nH,
        kh_results.GammaxH,
        kh_results.TH,
        kh_results.qxH_total,
        kh_results.NetHSource,
        kh_results.Sion,
        kh_results.QH_total,
        kh_results.SideWallH,
        Lyman,
        Balmer,

        GammaHLim,
    )

    return results
end
