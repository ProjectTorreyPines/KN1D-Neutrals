using Test
using JSON
using KN1DJl

const KINETIC_H2_TEST_CONFIG = joinpath(@__DIR__, "kinetic_h2_test_config.json")

function _write_kinetic_h2_test_config(; simple_cx::Bool=true)::Nothing
    open(KINETIC_H2_TEST_CONFIG, "w") do io
        JSON.print(io, Dict(
            "kinetic_h" => Dict(
                "mesh_size" => 6,
                "ion_rate" => "janev",
                "ci_test" => false,
                "alpha_cx_test" => false,
                "grid_fctr" => 0.3,
                "extra_energy_bins_eV" => Float64[],
            ),
            "kinetic_h2" => Dict(
                "mesh_size" => 5,
                "grid_fctr" => 0.3,
                "extra_energy_bins_eV" => Float64[],
                "ci_test" => false,
                "alpha_cx_test" => false,
            ),
            "collisions" => Dict(
                "H2_H_EL" => true,
                "H2_H2_EL" => true,
                "H2_P_EL" => true,
                "H2_P_CX" => true,
                "H_H_EL" => true,
                "H_P_EL" => true,
                "H_P_CX" => true,
                "SIMPLE_CX" => simple_cx,
            ),
        ))
    end
    return nothing
end

@testset "KineticH2 Constructor Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 parity requires python with numpy/scipy"
    else
        _write_kinetic_h2_test_config()
        config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h2 import KineticH2

            x = np.linspace(0.0, 0.2, 5)
            Ti = np.full(5, 2.0)
            Te = np.full(5, 3.0)
            n = np.full(5, 1.0e20)
            # Keep side-wall terms off here: Python kinetic_h2 has a known
            # broadcast bug in gamma_wall construction when PipeDia > 0.
            pipe = np.zeros(5)
            mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
            fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
            fH2BC[:, mesh.vx > 0] = 1.0
            NuLoss = np.linspace(0.0, 2.0, mesh.x.size)
            SH2 = np.zeros(mesh.x.size)
            kh2 = KineticH2(
                mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20, NuLoss, SH2,
                sawada=False, compute_h_source=False, ni_correct=True,
                truncate=1.0e-3, max_gen=10, config_path=$config_literal,
            )
            print(json.dumps({
                "nvr": kh2.nvr,
                "nvx": kh2.nvx,
                "nx": kh2.nx,
                "vth": float(kh2.vth),
                "fH2BC_input_sum": float(np.sum(kh2.fH2BC_input)),
                "fH2BC_input": kh2.fH2BC_input.tolist(),
                "Eaxis": kh2.Eaxis.tolist(),
                "dEaxis": kh2.dEaxis.tolist(),
                "dvr_vol": kh2.dvr_vol.tolist(),
                "dvr_vol_h_order": kh2.dvr_vol_h_order.tolist(),
                "dvx": kh2.dvx.tolist(),
                "vx_pos": kh2.vx_pos.tolist(),
                "vx_neg": kh2.vx_neg.tolist()
            }))
            """
        )

        x = collect(range(0.0, 0.2; length=5))
        mesh = KineticMesh(
            "h2",
            1,
            x,
            fill(2.0, 5),
            fill(3.0, 5),
            fill(1.0e20, 5),
            zeros(5);
            E0=[0.003, 0.03],
            config_path=KINETIC_H2_TEST_CONFIG,
        )
        fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
        @inbounds for j in eachindex(mesh.vx)
            if mesh.vx[j] > 0.0
                fH2BC[:, j] .= 1.0
            end
        end
        kh2 = KineticH2(
            mesh,
            1,
            zeros(length(mesh.x)),
            fH2BC,
            2.5e20,
            collect(range(0.0, 2.0; length=length(mesh.x))),
            zeros(length(mesh.x));
            sawada=false,
            compute_h_source=false,
            ni_correct=true,
            truncate=1.0e-3,
            max_gen=10,
            config_path=KINETIC_H2_TEST_CONFIG,
        )

        @test kh2.nvr == Int(py["nvr"])
        @test kh2.nvx == Int(py["nvx"])
        @test kh2.nx == Int(py["nx"])
        @test kh2.vth ≈ Float64(py["vth"]) rtol=1e-6
        @test sum(kh2.fH2BC_input) ≈ Float64(py["fH2BC_input_sum"]) rtol=1e-6
        @test kh2.fH2BC_input ≈ _to_float_matrix(py["fH2BC_input"]) rtol=1e-6
        @test kh2.Eaxis ≈ _to_float_vector(py["Eaxis"]) rtol=1e-13
        @test kh2.dEaxis ≈ _to_float_vector(py["dEaxis"]) rtol=1e-13
        @test kh2.dvr_vol ≈ _to_float_vector(py["dvr_vol"]) rtol=1e-13
        @test kh2.dvr_vol_h_order ≈ _to_float_vector(py["dvr_vol_h_order"]) rtol=1e-13
        @test kh2.dvx ≈ _to_float_vector(py["dvx"]) rtol=1e-13
        @test kh2.vx_pos == (Int.(py["vx_pos"]) .+ 1)
        @test kh2.vx_neg == (Int.(py["vx_neg"]) .+ 1)
    end
end

@testset "KineticH2 Initial Static Internals" begin
    _write_kinetic_h2_test_config()
    x = collect(range(0.0, 0.2; length=5))
    mesh = KineticMesh(
        "h2",
        1,
        x,
        fill(2.0, 5),
        fill(3.0, 5),
        fill(1.0e20, 5),
        fill(0.1, 5);
        E0=[0.003, 0.03],
        config_path=KINETIC_H2_TEST_CONFIG,
    )
    fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
    @inbounds for j in eachindex(mesh.vx)
        if mesh.vx[j] > 0.0
            fH2BC[:, j] .= 1.0
        end
    end

    kh2 = KineticH2(
        mesh,
        1,
        zeros(length(mesh.x)),
        fH2BC,
        2.5e20,
        zeros(length(mesh.x)),
        ones(length(mesh.x));
        sawada=false,
        truncate=1.0e-3,
        max_gen=10,
        config_path=KINETIC_H2_TEST_CONFIG,
        initialize_static=true,
    )

    @test size(kh2.internal.vr2vx2) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(kh2.internal.vr2vx_vxi2) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(kh2.internal.EH2_P) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(kh2.internal.fi_hat) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(kh2.internal.fw_hat) == (kh2.nvr, kh2.nvx)
    @test size(kh2.internal.sigv) == (kh2.nx, 11)
    @test length(kh2.internal.Alpha_Loss) == kh2.nx
    @test all(isfinite, kh2.internal.Alpha_Loss)
    @test size(kh2.internal.vx_vx) == (kh2.nvx, kh2.nvx)
    @test size(kh2.internal.Vr2pidVrdVx) == (kh2.nvr, kh2.nvx)
    @test kh2.internal.v_v2 === nothing
    @test kh2.internal.v_v === nothing
    @test kh2.internal.vr2_vx2 === nothing

    nvel = kh2.nvr * kh2.nvx
    @test size(kh2.internal.SIG_CX) == (nvel, nvel)
    @test size(kh2.internal.SIG_H2_H2) == (nvel, nvel)
    @test size(kh2.internal.SIG_H2_H) == (nvel, nvel)
    @test size(kh2.internal.SIG_H2_P) == (nvel, nvel)
    @test all(isfinite, kh2.internal.SIG_CX)
    @test all(isfinite, kh2.internal.SIG_H2_H2)
    @test all(isfinite, kh2.internal.SIG_H2_H)
    @test all(isfinite, kh2.internal.SIG_H2_P)
    @test any(!iszero, kh2.internal.SIG_CX)
    @test any(!iszero, kh2.internal.SIG_H2_H2)
    @test any(!iszero, kh2.internal.SIG_H2_H)
    @test any(!iszero, kh2.internal.SIG_H2_P)

    @inbounds for j2 in 1:kh2.nvx, j1 in 1:kh2.nvx
        @test kh2.internal.vx_vx[j1, j2] == kh2.mesh.vx[j1] - kh2.mesh.vx[j2]
    end
    @inbounds for j in 1:kh2.nvx, i in 1:kh2.nvr
        @test kh2.internal.Vr2pidVrdVx[i, j] == kh2.dvr_vol[i] * kh2.dvx[j]
    end
    @test run_procedure isa Function
end

@testset "KineticH2 Alpha_CX Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 Alpha_CX parity requires python with numpy/scipy"
    else
        for simple_cx in (true, false)
            _write_kinetic_h2_test_config(; simple_cx=simple_cx)
            config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

            py = _python_json(
                """
                import sys, json, numpy as np
                sys.path.insert(0, '.')
                from KN1DPy.kinetic_mesh import KineticMesh
                from KN1DPy.kinetic_h2 import KineticH2

                x = np.linspace(0.0, 0.2, 5)
                Ti = np.full(5, 2.0)
                Te = np.linspace(2.5, 4.5, 5)
                n = np.linspace(0.8e20, 1.2e20, 5)
                pipe = np.full(5, 0.1)
                mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
                fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
                fH2BC[:, mesh.vx > 0] = 1.0
                NuLoss = np.linspace(0.0, 2.0, mesh.x.size)
                SH2 = np.zeros(mesh.x.size)
                nHP = np.linspace(0.2e18, 1.0e18, mesh.x.size)
                THP = np.linspace(1.0, 5.0, mesh.x.size)
                kh2 = KineticH2(
                    mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20, NuLoss, SH2,
                    sawada=False, compute_h_source=False, ni_correct=True,
                    truncate=1.0e-3, max_gen=10, config_path=$config_literal,
                )
                kh2._compute_alpha_cx(nHP, THP)
                print(json.dumps({
                    "Alpha_CX": kh2.Internal.Alpha_CX.tolist(),
                    "fHp_hat": kh2.Internal.fHp_hat.tolist()
                }))
                """
            )

            x = collect(range(0.0, 0.2; length=5))
            mesh = KineticMesh(
                "h2",
                1,
                x,
                fill(2.0, 5),
                collect(range(2.5, 4.5; length=5)),
                collect(range(0.8e20, 1.2e20; length=5)),
                fill(0.1, 5);
                E0=[0.003, 0.03],
                config_path=KINETIC_H2_TEST_CONFIG,
            )
            fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
            @inbounds for j in eachindex(mesh.vx)
                if mesh.vx[j] > 0.0
                    fH2BC[:, j] .= 1.0
                end
            end

            kh2 = KineticH2(
                mesh,
                1,
                zeros(length(mesh.x)),
                fH2BC,
                2.5e20,
                collect(range(0.0, 2.0; length=length(mesh.x))),
                zeros(length(mesh.x));
                sawada=false,
                compute_h_source=false,
                ni_correct=true,
                truncate=1.0e-3,
                max_gen=10,
                config_path=KINETIC_H2_TEST_CONFIG,
                initialize_static=true,
            )

            nHP = collect(range(0.2e18, 1.0e18; length=length(mesh.x)))
            THP = collect(range(1.0, 5.0; length=length(mesh.x)))
            KN1DJl._compute_alpha_cx!(kh2, nHP, THP)

            @test kh2.internal.fHp_hat ≈ _to_float_array3(py["fHp_hat"]) rtol=1e-10 atol=1e-12
            # The simple <sigma v> branch routes through independent Julia/Python
            # polynomial evaluators and differs at ~1e-7 relative; the direct
            # SIG_CX branch is a kernel-orientation regression and should stay tight.
            alpha_rtol = simple_cx ? 2e-7 : 1e-10
            @test kh2.internal.Alpha_CX ≈ _to_float_array3(py["Alpha_CX"]) rtol=alpha_rtol atol=1e-12
        end
    end
end

@testset "KineticH2 fH Moment Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 fH moment parity requires python with numpy/scipy"
    else
        _write_kinetic_h2_test_config()
        config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h2 import KineticH2

            x = np.linspace(0.0, 0.2, 5)
            Ti = np.full(5, 2.0)
            Te = np.linspace(2.5, 4.5, 5)
            n = np.linspace(0.8e20, 1.2e20, 5)
            # Keep side-wall terms off here: Python kinetic_h2 has a known
            # broadcast bug in gamma_wall construction when PipeDia > 0.
            pipe = np.zeros(5)
            mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
            fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
            fH2BC[:, mesh.vx > 0] = 1.0
            kh2 = KineticH2(
                mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20,
                np.linspace(0.0, 2.0, mesh.x.size), np.zeros(mesh.x.size),
                sawada=False, compute_h_source=False, ni_correct=True,
                truncate=1.0e-3, max_gen=10, config_path=$config_literal,
            )

            i = np.arange(mesh.vr.size)[:, None, None]
            j = np.arange(mesh.vx.size)[None, :, None]
            k = np.arange(mesh.x.size)[None, None, :]
            fH = 0.2 + 0.01*(i + 1) + 0.02*(j + 1) + 0.03*(k + 1)
            fH[:, :, 1] = 0.0

            kh2.H_Moments.nH = np.zeros(kh2.nx)
            kh2.H_Moments.VxH = np.zeros(kh2.nx)
            kh2.H_Moments.TH = np.full(kh2.nx, 1.0)
            kh2._compute_fh_moments(fH)

            print(json.dumps({
                "fH": fH.tolist(),
                "nH": kh2.H_Moments.nH.tolist(),
                "VxH": kh2.H_Moments.VxH.tolist(),
                "TH": kh2.H_Moments.TH.tolist()
            }))
            """
        )

        x = collect(range(0.0, 0.2; length=5))
        mesh = KineticMesh(
            "h2",
            1,
            x,
            fill(2.0, 5),
            collect(range(2.5, 4.5; length=5)),
            collect(range(0.8e20, 1.2e20; length=5)),
            zeros(5);
            E0=[0.003, 0.03],
            config_path=KINETIC_H2_TEST_CONFIG,
        )
        fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
        @inbounds for j in eachindex(mesh.vx)
            if mesh.vx[j] > 0.0
                fH2BC[:, j] .= 1.0
            end
        end
        kh2 = KineticH2(
            mesh,
            1,
            zeros(length(mesh.x)),
            fH2BC,
            2.5e20,
            collect(range(0.0, 2.0; length=length(mesh.x))),
            zeros(length(mesh.x));
            sawada=false,
            compute_h_source=false,
            ni_correct=true,
            truncate=1.0e-3,
            max_gen=10,
            config_path=KINETIC_H2_TEST_CONFIG,
        )

        fH = _to_float_array3(py["fH"])
        KN1DJl._compute_fh_moments!(kh2, fH)

        @test kh2.h_moments.nH ≈ _to_float_vector(py["nH"]) rtol=1e-13 atol=1e-12
        @test kh2.h_moments.VxH ≈ _to_float_vector(py["VxH"]) rtol=2e-7 atol=1e-12
        @test kh2.h_moments.TH ≈ _to_float_vector(py["TH"]) rtol=4e-7 atol=1e-12
        @test kh2.h_moments.VxH[2] == 0.0
        @test kh2.h_moments.TH[2] == 1.0
        @test_throws DimensionMismatch KN1DJl._compute_fh_moments!(kh2, zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx + 1))
    end
end

@testset "KineticH2 Swall Formula" begin
    _write_kinetic_h2_test_config()
    x = collect(range(0.0, 0.2; length=5))
    mesh = KineticMesh(
        "h2",
        1,
        x,
        fill(2.0, 5),
        fill(3.0, 5),
        fill(1.0e20, 5),
        fill(0.1, 5);
        E0=[0.003, 0.03],
        config_path=KINETIC_H2_TEST_CONFIG,
    )
    fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
    @inbounds for j in eachindex(mesh.vx)
        if mesh.vx[j] > 0.0
            fH2BC[:, j] .= 1.0
        end
    end
    kh2 = KineticH2(
        mesh,
        1,
        zeros(length(mesh.x)),
        fH2BC,
        2.5e20,
        zeros(length(mesh.x)),
        ones(length(mesh.x));
        sawada=false,
        truncate=1.0e-3,
        max_gen=10,
        config_path=KINETIC_H2_TEST_CONFIG,
        initialize_static=true,
    )

    fH2G = Array{Float64,3}(undef, kh2.nvr, kh2.nvx, kh2.nx)
    gamma_wall = Array{Float64,3}(undef, kh2.nvr, kh2.nvx, kh2.nx)
    @inbounds for k in 1:kh2.nx, j in 1:kh2.nvx, i in 1:kh2.nvr
        fH2G[i, j, k] = 0.1 + 0.01*i + 0.02*j + 0.03*k
        gamma_wall[i, j, k] = 0.05 + 0.001*i + 0.002*j + 0.003*k
    end

    Swall = KN1DJl._compute_swall(kh2, fH2G, gamma_wall)
    @inbounds for k in 1:kh2.nx
        wall_source = 0.0
        for j in 1:kh2.nvx, i in 1:kh2.nvr
            wall_source += kh2.dvr_vol[i] * gamma_wall[i, j, k] * fH2G[i, j, k] * kh2.dvx[j]
        end
        @test Swall[:, :, k] ≈ kh2.internal.fw_hat .* wall_source rtol=1e-13 atol=1e-12
    end

    @test all(iszero, KN1DJl._compute_swall(kh2, fH2G, zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)))
    @test_throws DimensionMismatch KN1DJl._compute_swall(kh2, fH2G, zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx + 1))
end

@testset "KineticH2 Beta_CX Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 Beta_CX parity requires python with numpy/scipy"
    else
        for simple_cx in (true, false)
            _write_kinetic_h2_test_config(; simple_cx=simple_cx)
            config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

            py = _python_json(
                """
                import sys, json, numpy as np
                sys.path.insert(0, '.')
                from KN1DPy.kinetic_mesh import KineticMesh
                from KN1DPy.kinetic_h2 import KineticH2

                x = np.linspace(0.0, 0.2, 5)
                Ti = np.full(5, 2.0)
                Te = np.linspace(2.5, 4.5, 5)
                n = np.linspace(0.8e20, 1.2e20, 5)
                pipe = np.full(5, 0.1)
                mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
                fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
                fH2BC[:, mesh.vx > 0] = 1.0
                kh2 = KineticH2(
                    mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20,
                    np.linspace(0.0, 2.0, mesh.x.size), np.zeros(mesh.x.size),
                    sawada=False, compute_h_source=False, ni_correct=True,
                    truncate=1.0e-3, max_gen=10, config_path=$config_literal,
                )
                nHP = np.linspace(0.2e18, 1.0e18, mesh.x.size)
                THP = np.linspace(1.0, 5.0, mesh.x.size)
                kh2._compute_alpha_cx(nHP, THP)

                i = np.arange(mesh.vr.size)[:, None, None]
                j = np.arange(mesh.vx.size)[None, :, None]
                k = np.arange(mesh.x.size)[None, None, :]
                fH2G = 0.05 + 0.005*(i + 1) + 0.007*(j + 1) + 0.011*(k + 1)
                Beta_CX = kh2._compute_beta_cx(fH2G, nHP)
                print(json.dumps({
                    "fH2G": fH2G.tolist(),
                    "Beta_CX": Beta_CX.tolist()
                }))
                """
            )

            x = collect(range(0.0, 0.2; length=5))
            mesh = KineticMesh(
                "h2",
                1,
                x,
                fill(2.0, 5),
                collect(range(2.5, 4.5; length=5)),
                collect(range(0.8e20, 1.2e20; length=5)),
                fill(0.1, 5);
                E0=[0.003, 0.03],
                config_path=KINETIC_H2_TEST_CONFIG,
            )
            fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
            @inbounds for j in eachindex(mesh.vx)
                if mesh.vx[j] > 0.0
                    fH2BC[:, j] .= 1.0
                end
            end
            kh2 = KineticH2(
                mesh,
                1,
                zeros(length(mesh.x)),
                fH2BC,
                2.5e20,
                collect(range(0.0, 2.0; length=length(mesh.x))),
                zeros(length(mesh.x));
                sawada=false,
                compute_h_source=false,
                ni_correct=true,
                truncate=1.0e-3,
                max_gen=10,
                config_path=KINETIC_H2_TEST_CONFIG,
                initialize_static=true,
            )

            nHP = collect(range(0.2e18, 1.0e18; length=length(mesh.x)))
            THP = collect(range(1.0, 5.0; length=length(mesh.x)))
            KN1DJl._compute_alpha_cx!(kh2, nHP, THP)
            Beta_CX = KN1DJl._compute_beta_cx(kh2, _to_float_array3(py["fH2G"]), nHP)

            beta_rtol = simple_cx ? 3e-7 : 1e-10
            @test Beta_CX ≈ _to_float_array3(py["Beta_CX"]) rtol=beta_rtol atol=1e-12
            @test_throws DimensionMismatch KN1DJl._compute_beta_cx(kh2, zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx + 1), nHP)
            @test_throws DimensionMismatch KN1DJl._compute_beta_cx(kh2, _to_float_array3(py["fH2G"]), [1.0])
        end
    end
end

@testset "KineticH2 MH Values Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 MH parity requires python with numpy/scipy"
    else
        _write_kinetic_h2_test_config()
        config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h2 import KineticH2

            x = np.linspace(0.0, 0.2, 5)
            Ti = np.full(5, 2.0)
            Te = np.linspace(2.5, 4.5, 5)
            n = np.linspace(0.8e20, 1.2e20, 5)
            # Keep side-wall terms off here: Python kinetic_h2 has a known
            # broadcast bug in gamma_wall construction when PipeDia > 0.
            pipe = np.zeros(5)
            mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
            fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
            fH2BC[:, mesh.vx > 0] = 1.0
            kh2 = KineticH2(
                mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20,
                np.linspace(0.0, 2.0, mesh.x.size), np.zeros(mesh.x.size),
                sawada=False, compute_h_source=False, ni_correct=True,
                truncate=1.0e-3, max_gen=10, config_path=$config_literal,
            )

            i = np.arange(mesh.vr.size)[:, None, None]
            j = np.arange(mesh.vx.size)[None, :, None]
            k = np.arange(mesh.x.size)[None, None, :]
            fH2G = 0.08 + 0.004*(i + 1) + 0.006*(j + 1) + 0.009*(k + 1)
            nH2 = np.zeros(mesh.x.size)
            for kk in range(mesh.x.size):
                nH2[kk] = np.sum(kh2.dvr_vol * (fH2G[:, :, kk] @ kh2.dvx))

            kh2.H_Moments.VxH = np.linspace(-750.0, 650.0, mesh.x.size)
            kh2.H_Moments.TH = np.linspace(0.7, 2.3, mesh.x.size)
            vals = kh2._compute_mh_values(fH2G, nH2)
            print(json.dumps({
                "fH2G": fH2G.tolist(),
                "nH2": nH2.tolist(),
                "VxH": kh2.H_Moments.VxH.tolist(),
                "TH": kh2.H_Moments.TH.tolist(),
                "H2_H2": vals.H2_H2.tolist(),
                "H2_P": vals.H2_P.tolist(),
                "H2_H": vals.H2_H.tolist()
            }))
            """
        )

        x = collect(range(0.0, 0.2; length=5))
        mesh = KineticMesh(
            "h2",
            1,
            x,
            fill(2.0, 5),
            collect(range(2.5, 4.5; length=5)),
            collect(range(0.8e20, 1.2e20; length=5)),
            zeros(5);
            E0=[0.003, 0.03],
            config_path=KINETIC_H2_TEST_CONFIG,
        )
        fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
        @inbounds for j in eachindex(mesh.vx)
            if mesh.vx[j] > 0.0
                fH2BC[:, j] .= 1.0
            end
        end
        kh2 = KineticH2(
            mesh,
            1,
            zeros(length(mesh.x)),
            fH2BC,
            2.5e20,
            collect(range(0.0, 2.0; length=length(mesh.x))),
            zeros(length(mesh.x));
            sawada=false,
            compute_h_source=false,
            ni_correct=true,
            truncate=1.0e-3,
            max_gen=10,
            config_path=KINETIC_H2_TEST_CONFIG,
        )
        kh2.h_moments.VxH = _to_float_vector(py["VxH"])
        kh2.h_moments.TH = _to_float_vector(py["TH"])

        vals = KN1DJl._compute_mh_values(kh2, _to_float_array3(py["fH2G"]), _to_float_vector(py["nH2"]))

        @test vals.H2_H2 ≈ _to_float_array3(py["H2_H2"]) rtol=5e-7 atol=1e-12
        @test vals.H2_P ≈ _to_float_array3(py["H2_P"]) rtol=5e-7 atol=1e-12
        @test vals.H2_H ≈ _to_float_array3(py["H2_H"]) rtol=5e-7 atol=1e-12
        @test_throws DimensionMismatch KN1DJl._compute_mh_values(kh2, zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx + 1), _to_float_vector(py["nH2"]))
        @test_throws DimensionMismatch KN1DJl._compute_mh_values(kh2, _to_float_array3(py["fH2G"]), [1.0])
    end
end

@testset "KineticH2 Collision Frequency And Mesh Coefficients Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 collision/mesh parity requires python with numpy/scipy"
    else
        _write_kinetic_h2_test_config()
        config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h2 import KineticH2, CollisionType

            x = np.linspace(0.0, 0.2, 5)
            Ti = np.full(5, 2.0)
            Te = np.linspace(2.5, 4.5, 5)
            n = np.linspace(0.8e20, 1.2e20, 5)
            # Keep side-wall terms off here: Python kinetic_h2 has a known
            # broadcast bug in gamma_wall construction when PipeDia > 0.
            pipe = np.zeros(5)
            mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
            fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
            fH2BC[:, mesh.vx > 0] = 1.0
            kh2 = KineticH2(
                mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20,
                np.linspace(0.0, 2.0, mesh.x.size), np.ones(mesh.x.size),
                sawada=False, compute_h_source=False, ni_correct=True,
                truncate=1.0e-3, max_gen=10, config_path=$config_literal,
            )
            i = np.arange(mesh.vr.size)[:, None, None]
            j = np.arange(mesh.vx.size)[None, :, None]
            k = np.arange(mesh.x.size)[None, None, :]
            kh2.Internal.Alpha_CX = 0.01 + 0.001*(i + 1) + 0.002*(j + 1) + 0.003*(k + 1)
            kh2.Internal.Alpha_Loss = np.linspace(0.02, 0.06, mesh.x.size)
            gamma_wall = 1.0e-4 + 1.0e-5*(i + 1) + 2.0e-5*(j + 1) + 3.0e-5*(k + 1)
            collision_freqs = CollisionType(
                np.linspace(0.02, 0.06, mesh.x.size),
                np.linspace(0.03, 0.07, mesh.x.size),
                np.linspace(0.01, 0.05, mesh.x.size),
            )
            alpha_c = kh2._compute_collision_frequency(collision_freqs, gamma_wall)
            SH2 = np.linspace(0.1, 0.5, mesh.x.size)
            coeffs = kh2._compute_mesh_equation_coefficients(alpha_c, SH2)
            print(json.dumps({
                "gamma_wall": gamma_wall.tolist(),
                "SH2": SH2.tolist(),
                "Alpha_CX": kh2.Internal.Alpha_CX.tolist(),
                "Alpha_Loss": kh2.Internal.Alpha_Loss.tolist(),
                "omega_h2_h2": collision_freqs.H2_H2.tolist(),
                "omega_h2_p": collision_freqs.H2_P.tolist(),
                "omega_h2_h": collision_freqs.H2_H.tolist(),
                "alpha_c": alpha_c.tolist(),
                "Max_dx": kh2.Errors.Max_dx.tolist(),
                "A": coeffs.A.tolist(),
                "B": coeffs.B.tolist(),
                "C": coeffs.C.tolist(),
                "D": coeffs.D.tolist(),
                "F": coeffs.F.tolist(),
                "G": coeffs.G.tolist()
            }))
            """
        )

            x = collect(range(0.0, 0.2; length=5))
        mesh = KineticMesh(
            "h2",
            1,
            x,
            fill(2.0, 5),
            collect(range(2.5, 4.5; length=5)),
            collect(range(0.8e20, 1.2e20; length=5)),
            zeros(5);
            E0=[0.003, 0.03],
            config_path=KINETIC_H2_TEST_CONFIG,
        )
        fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
        @inbounds for j in eachindex(mesh.vx)
            if mesh.vx[j] > 0.0
                fH2BC[:, j] .= 1.0
            end
        end
        kh2 = KineticH2(
            mesh,
            1,
            zeros(length(mesh.x)),
            fH2BC,
            2.5e20,
            collect(range(0.0, 2.0; length=length(mesh.x))),
            ones(length(mesh.x));
            sawada=false,
            compute_h_source=false,
            ni_correct=true,
            truncate=1.0e-3,
            max_gen=10,
            config_path=KINETIC_H2_TEST_CONFIG,
            initialize_static=true,
        )
        kh2.internal.Alpha_CX = _to_float_array3(py["Alpha_CX"])
        kh2.internal.Alpha_Loss = _to_float_vector(py["Alpha_Loss"])

        collision_freqs = KH2CollisionType(
            _to_float_vector(py["omega_h2_h2"]),
            _to_float_vector(py["omega_h2_p"]),
            _to_float_vector(py["omega_h2_h"]),
        )
        gamma_wall = _to_float_array3(py["gamma_wall"])
        alpha_c = KN1DJl._compute_collision_frequency(kh2, collision_freqs, gamma_wall)
        coeffs = KN1DJl._compute_mesh_equation_coefficients(kh2, alpha_c, _to_float_vector(py["SH2"]))

        @test alpha_c ≈ _to_float_array3(py["alpha_c"]) rtol=1e-12 atol=1e-12
        @test kh2.errors.Max_dx ≈ _to_float_vector(py["Max_dx"]) rtol=1e-12 atol=1e-12
        @test coeffs.A ≈ _to_float_array3(py["A"]) rtol=5e-5 atol=1e-12
        @test coeffs.B ≈ _to_float_array3(py["B"]) rtol=5e-5 atol=1e-12
        @test coeffs.C ≈ _to_float_array3(py["C"]) rtol=5e-5 atol=1e-12
        @test coeffs.D ≈ _to_float_array3(py["D"]) rtol=5e-5 atol=1e-12
        @test coeffs.F ≈ _to_float_array3(py["F"]) rtol=5e-5 atol=1e-12
        @test coeffs.G ≈ _to_float_array3(py["G"]) rtol=5e-5 atol=1e-12
        @test_throws DimensionMismatch KN1DJl._compute_collision_frequency(kh2, collision_freqs, zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx + 1))
        @test_throws DimensionMismatch KN1DJl._compute_mesh_equation_coefficients(kh2, alpha_c, [1.0])
    end
end

@testset "KineticH2 Run Generations Smoke" begin
    _write_kinetic_h2_test_config()
    x = collect(range(0.0, 0.2; length=5))
    mesh = KineticMesh(
        "h2",
        1,
        x,
        fill(2.0, 5),
        fill(3.0, 5),
        fill(1.0e20, 5),
        fill(0.1, 5);
        E0=[0.003, 0.03],
        config_path=KINETIC_H2_TEST_CONFIG,
    )
    fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
    @inbounds for j in eachindex(mesh.vx)
        if mesh.vx[j] > 0.0
            fH2BC[:, j] .= 1.0
        end
    end
    kh2 = KineticH2(
        mesh,
        1,
        zeros(length(mesh.x)),
        fH2BC,
        2.5e20,
        zeros(length(mesh.x)),
        ones(length(mesh.x));
        sawada=false,
        truncate=1.0e9,
        max_gen=4,
        config_path=KINETIC_H2_TEST_CONFIG,
        initialize_static=true,
    )

    fH2 = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    nH2 = zeros(Float64, kh2.nx)
    fH2G = fill(1.0e-4, kh2.nvr, kh2.nvx, kh2.nx)
    NH2G = zeros(Float64, kh2.nx, kh2.max_gen + 1)
    @inbounds for k in 1:kh2.nx
        s = 0.0
        for j in 1:kh2.nvx, i in 1:kh2.nvr
            s += kh2.dvr_vol[i] * fH2G[i, j, k] * kh2.dvx[j]
        end
        NH2G[k, 1] = s
    end

    kh2.internal.fHp_hat = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    kh2.internal.Alpha_CX = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    kh2.h_moments.VxH = zeros(Float64, kh2.nx)
    kh2.h_moments.TH = fill(1.0, kh2.nx)

    gamma_wall = fill(0.01, kh2.nvr, kh2.nvx, kh2.nx)
    A = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    B = fill(0.1, kh2.nvr, kh2.nvx, kh2.nx)
    C = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    D = fill(0.1, kh2.nvr, kh2.nvx, kh2.nx)
    F = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    G = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    meq = KH2MeshEqCoefficients(A, B, C, D, F, G)
    freqs = KH2CollisionType(zeros(kh2.nx), zeros(kh2.nx), zeros(kh2.nx))

    fH2_out, nH2_out, fH2G_out, NH2G_out, Swall_sum, Beta_CX_sum, m_sums, igen =
        KN1DJl._run_generations(kh2, fH2, nH2, fH2G, NH2G, zeros(kh2.nx), gamma_wall, meq, freqs, true)

    @test igen == 1
    @test size(fH2_out) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test length(nH2_out) == kh2.nx
    @test size(fH2G_out) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(NH2G_out) == (kh2.nx, kh2.max_gen + 1)
    @test any(!iszero, Swall_sum)
    @test all(iszero, Beta_CX_sum)
    @test m_sums isa KH2CollisionType
    @test all(nH2_out .>= 0.0)
end

@testset "KineticH2 Iteration Scheme Smoke" begin
    _write_kinetic_h2_test_config()
    x = collect(range(0.0, 0.2; length=5))
    mesh = KineticMesh(
        "h2",
        1,
        x,
        fill(2.0, 5),
        fill(3.0, 5),
        fill(1.0e20, 5),
        fill(0.1, 5);
        E0=[0.003, 0.03],
        config_path=KINETIC_H2_TEST_CONFIG,
    )
    fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
    @inbounds for j in eachindex(mesh.vx)
        if mesh.vx[j] > 0.0
            fH2BC[:, j] .= 1.0
        end
    end
    kh2 = KineticH2(
        mesh,
        1,
        zeros(length(mesh.x)),
        fH2BC,
        2.5e20,
        fill(1.0e20, length(mesh.x)),
        fill(1.0e15, length(mesh.x));
        sawada=false,
        truncate=1.0e9,
        max_gen=4,
        config_path=KINETIC_H2_TEST_CONFIG,
        initialize_static=true,
    )

    fH2 = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    @inbounds for j in kh2.vx_pos, i in 1:kh2.nvr
        fH2[i, j, 1] = kh2.fH2BC_input[i, j]
    end
    nH2 = fill(1.0, kh2.nx)
    nHP = fill(1.0e18, kh2.nx)
    THP = fill(3.0, kh2.nx)
    SH2 = fill(1.0e15, kh2.nx)
    gamma_wall = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)

    kh2.internal.Alpha_H2_P = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    kh2.internal.Alpha_H2_H = zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx)
    kh2.h_moments.VxH = zeros(Float64, kh2.nx)
    kh2.h_moments.TH = fill(1.0, kh2.nx)

    fH2_out, alpha_c, Beta_CX_sum, Swall_sum, collision_freqs, m_sums =
        KN1DJl._run_iteration_scheme(kh2, fH2, nH2, nHP, THP, SH2, gamma_wall, true, false)

    @test size(fH2_out) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(alpha_c) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(Beta_CX_sum) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(Swall_sum) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test collision_freqs isa KH2CollisionType
    @test m_sums isa KH2CollisionType
    @test all(isfinite, fH2_out)
    @test all(isfinite, alpha_c)
end

@testset "KineticH2 Compile Results Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 compile-result parity requires python with numpy/scipy"
    else
        _write_kinetic_h2_test_config()
        config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h2 import KineticH2, CollisionType

            x = np.linspace(0.0, 0.2, 5)
            Ti = np.full(5, 2.0)
            Te = np.linspace(2.5, 4.5, 5)
            n = np.linspace(0.8e20, 1.2e20, 5)
            pipe = np.full(5, 0.1)
            mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
            fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
            fH2BC[:, mesh.vx > 0] = 1.0
            kh2 = KineticH2(
                mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20,
                np.linspace(0.0, 2.0, mesh.x.size), np.zeros(mesh.x.size),
                sawada=False, compute_h_source=False, ni_correct=True,
                truncate=1.0e-3, max_gen=10, config_path=$config_literal,
            )

            i = np.arange(mesh.vr.size)[:, None, None]
            j = np.arange(mesh.vx.size)[None, :, None]
            k = np.arange(mesh.x.size)[None, None, :]
            fH2 = 0.08 + 0.004*(i + 1) + 0.006*(j + 1) + 0.0003*(k + 1)
            SH2 = np.linspace(0.1, 0.5, mesh.x.size)
            gamma_wall = 1.0e-4 + 1.0e-5*(i + 1) + 2.0e-5*(j + 1) + 3.0e-7*(k + 1)
            alpha_c = 0.02 + 0.001*(i + 1) + 0.002*(j + 1) + 0.0001*(k + 1)
            Beta_CX_sum = 1.0e-5 + 1.0e-6*(i + 1) + 2.0e-6*(j + 1) + 1.0e-8*(k + 1)
            Swall_sum = 2.0e-5 + 1.5e-6*(i + 1) + 1.0e-6*(j + 1) + 1.0e-8*(k + 1)
            kh2.Internal.Alpha_CX = 0.003 + 1.0e-4*(i + 1) + 2.0e-4*(j + 1) + 1.0e-6*(k + 1)
            collision_freqs = CollisionType(
                np.linspace(0.02, 0.06, mesh.x.size),
                np.linspace(0.03, 0.07, mesh.x.size),
                np.linspace(0.01, 0.05, mesh.x.size),
            )
            m_sums = CollisionType(
                0.01 + 0.001*(i + 1) + 0.0002*(j + 1) + 1.0e-6*(k + 1),
                0.02 + 0.0007*(i + 1) + 0.0003*(j + 1) + 1.0e-6*(k + 1),
                0.03 + 0.0005*(i + 1) + 0.0004*(j + 1) + 1.0e-6*(k + 1),
            )
            r = kh2._compile_results(fH2, SH2, gamma_wall, alpha_c, Beta_CX_sum, Swall_sum, collision_freqs, m_sums)
            print(json.dumps({
                "fH2": fH2.tolist(),
                "SH2": SH2.tolist(),
                "gamma_wall": gamma_wall.tolist(),
                "alpha_c": alpha_c.tolist(),
                "Beta_CX_sum": Beta_CX_sum.tolist(),
                "Swall_sum": Swall_sum.tolist(),
                "Alpha_CX": kh2.Internal.Alpha_CX.tolist(),
                "omega_h2_h2": collision_freqs.H2_H2.tolist(),
                "omega_h2_p": collision_freqs.H2_P.tolist(),
                "omega_h2_h": collision_freqs.H2_H.tolist(),
                "m_h2_h2": m_sums.H2_H2.tolist(),
                "m_h2_p": m_sums.H2_P.tolist(),
                "m_h2_h": m_sums.H2_H.tolist(),
                "nH2": r.nH2.tolist(),
                "GammaxH2": r.GammaxH2.tolist(),
                "VxH2": r.VxH2.tolist(),
                "pH2": r.pH2.tolist(),
                "TH2": r.TH2.tolist(),
                "qxH2": r.qxH2.tolist(),
                "qxH2_total": r.qxH2_total.tolist(),
                "Sloss": r.Sloss.tolist(),
                "QH2": r.QH2.tolist(),
                "RxH2": r.RxH2.tolist(),
                "QH2_total": r.QH2_total.tolist(),
                "WallH2": r.WallH2.tolist(),
                "NuE": r.NuE.tolist(),
                "NuDis": r.NuDis.tolist(),
                "AlbedoH2": float(r.AlbedoH2),
                "piH2_xx": kh2.Output.piH2_xx.tolist(),
                "piH2_yy": kh2.Output.piH2_yy.tolist()
            }))
            """
        )

        x = collect(range(0.0, 0.2; length=5))
        mesh = KineticMesh(
            "h2",
            1,
            x,
            fill(2.0, 5),
            collect(range(2.5, 4.5; length=5)),
            collect(range(0.8e20, 1.2e20; length=5)),
            fill(0.1, 5);
            E0=[0.003, 0.03],
            config_path=KINETIC_H2_TEST_CONFIG,
        )
        fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
        @inbounds for j in eachindex(mesh.vx)
            if mesh.vx[j] > 0.0
                fH2BC[:, j] .= 1.0
            end
        end
        kh2 = KineticH2(
            mesh,
            1,
            zeros(length(mesh.x)),
            fH2BC,
            2.5e20,
            collect(range(0.0, 2.0; length=length(mesh.x))),
            zeros(length(mesh.x));
            sawada=false,
            compute_h_source=false,
            ni_correct=true,
            truncate=1.0e-3,
            max_gen=10,
            config_path=KINETIC_H2_TEST_CONFIG,
            initialize_static=true,
        )

        kh2.internal.Alpha_CX = _to_float_array3(py["Alpha_CX"])
        collision_freqs = KH2CollisionType(
            _to_float_vector(py["omega_h2_h2"]),
            _to_float_vector(py["omega_h2_p"]),
            _to_float_vector(py["omega_h2_h"]),
        )
        m_sums = KH2CollisionType(
            _to_float_array3(py["m_h2_h2"]),
            _to_float_array3(py["m_h2_p"]),
            _to_float_array3(py["m_h2_h"]),
        )
        r = KN1DJl._compile_results(
            kh2,
            _to_float_array3(py["fH2"]),
            _to_float_vector(py["SH2"]),
            _to_float_array3(py["gamma_wall"]),
            _to_float_array3(py["alpha_c"]),
            _to_float_array3(py["Beta_CX_sum"]),
            _to_float_array3(py["Swall_sum"]),
            collision_freqs,
            m_sums,
        )

        @test r.nH2 ≈ _to_float_vector(py["nH2"]) rtol=1e-12 atol=1e-12
        # Mesh construction is independently ported, so quantities involving vx,
        # Ti, or fw_hat inherit tiny mesh-interpolation differences. These stay
        # well below the existing end-to-end parity envelope.
        @test r.GammaxH2 ≈ _to_float_vector(py["GammaxH2"]) rtol=2e-7 atol=1e-12
        @test r.VxH2 ≈ _to_float_vector(py["VxH2"]) rtol=2e-7 atol=1e-12
        @test r.pH2 ≈ _to_float_vector(py["pH2"]) rtol=1e-12 atol=1e-12
        @test r.TH2 ≈ _to_float_vector(py["TH2"]) rtol=1e-12 atol=1e-12
        @test r.qxH2 ≈ _to_float_vector(py["qxH2"]) rtol=1e-12 atol=1e-12
        @test r.qxH2_total ≈ _to_float_vector(py["qxH2_total"]) rtol=1e-12 atol=1e-12
        @test r.Sloss ≈ _to_float_vector(py["Sloss"]) rtol=2e-7 atol=1e-12
        @test r.QH2 ≈ _to_float_vector(py["QH2"]) rtol=1e-12 atol=1e-12
        @test r.RxH2 ≈ _to_float_vector(py["RxH2"]) rtol=1e-12 atol=1e-12
        @test r.QH2_total ≈ _to_float_vector(py["QH2_total"]) rtol=1e-12 atol=1e-12
        @test r.WallH2 ≈ _to_float_vector(py["WallH2"]) rtol=1e-12 atol=1e-12
        @test r.NuE ≈ _to_float_vector(py["NuE"]) rtol=1e-8 atol=1e-12
        @test r.NuDis ≈ _to_float_vector(py["NuDis"]) rtol=1e-8 atol=1e-12
        @test r.AlbedoH2 ≈ Float64(py["AlbedoH2"]) rtol=1e-12 atol=1e-12
        @test kh2.output.piH2_xx ≈ _to_float_vector(py["piH2_xx"]) rtol=1e-12 atol=1e-12
        @test kh2.output.piH2_yy ≈ _to_float_vector(py["piH2_yy"]) rtol=1e-12 atol=1e-12
    end
end

@testset "KineticH2 H Source Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 H-source parity requires python with numpy/scipy"
    else
        _write_kinetic_h2_test_config()
        config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h2 import KineticH2

            x = np.linspace(0.0, 0.2, 5)
            Ti = np.full(5, 2.0)
            Te = np.linspace(2.5, 4.5, 5)
            n = np.linspace(0.8e20, 1.2e20, 5)
            pipe = np.full(5, 0.1)
            mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
            fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
            fH2BC[:, mesh.vx > 0] = 1.0
            kh2 = KineticH2(
                mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20,
                np.linspace(0.0, 2.0, mesh.x.size), np.zeros(mesh.x.size),
                sawada=False, compute_h_source=True, ni_correct=True,
                truncate=1.0e-3, max_gen=10, config_path=$config_literal,
            )

            nH2 = np.linspace(1.0e16, 2.0e16, mesh.x.size)
            nHP = np.linspace(0.2e18, 1.0e18, mesh.x.size)
            THP = np.linspace(1.0, 5.0, mesh.x.size)
            SH2 = np.linspace(0.1e15, 0.5e15, mesh.x.size)
            TH2 = np.linspace(0.5, 1.5, mesh.x.size)
            GammaxH2 = np.linspace(1.0e18, 2.0e18, mesh.x.size)
            fSH, SH, SP, SHP, ESH = kh2._compute_h_source(nH2, nHP, THP, SH2, TH2, GammaxH2)

            print(json.dumps({
                "nH2": nH2.tolist(),
                "nHP": nHP.tolist(),
                "THP": THP.tolist(),
                "SH2": SH2.tolist(),
                "TH2": TH2.tolist(),
                "GammaxH2": GammaxH2.tolist(),
                "fSH": fSH.tolist(),
                "SH": SH.tolist(),
                "SP": SP.tolist(),
                "SHP": SHP.tolist(),
                "ESH": ESH.tolist()
            }))
            """
        )

        x = collect(range(0.0, 0.2; length=5))
        mesh = KineticMesh(
            "h2",
            1,
            x,
            fill(2.0, 5),
            collect(range(2.5, 4.5; length=5)),
            collect(range(0.8e20, 1.2e20; length=5)),
            fill(0.1, 5);
            E0=[0.003, 0.03],
            config_path=KINETIC_H2_TEST_CONFIG,
        )
        fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
        @inbounds for j in eachindex(mesh.vx)
            if mesh.vx[j] > 0.0
                fH2BC[:, j] .= 1.0
            end
        end
        kh2 = KineticH2(
            mesh,
            1,
            zeros(length(mesh.x)),
            fH2BC,
            2.5e20,
            collect(range(0.0, 2.0; length=length(mesh.x))),
            zeros(length(mesh.x));
            sawada=false,
            compute_h_source=true,
            ni_correct=true,
            truncate=1.0e-3,
            max_gen=10,
            config_path=KINETIC_H2_TEST_CONFIG,
            initialize_static=true,
        )

        fSH, SH, SP, SHP, ESH = KN1DJl._compute_h_source(
            kh2,
            _to_float_vector(py["nH2"]),
            _to_float_vector(py["nHP"]),
            _to_float_vector(py["THP"]),
            _to_float_vector(py["SH2"]),
            _to_float_vector(py["TH2"]),
            _to_float_vector(py["GammaxH2"]),
        )

        # Source distributions depend on the independently constructed kinetic
        # mesh and reaction-rate fits, so this follows the established H2 mesh
        # parity tolerance rather than bitwise equality.
        @test fSH ≈ _to_float_array3(py["fSH"]) rtol=2e-6 atol=1e-12
        @test SH ≈ _to_float_vector(py["SH"]) rtol=2e-6 atol=1e-12
        @test SP ≈ _to_float_vector(py["SP"]) rtol=2e-6 atol=1e-12
        @test SHP ≈ _to_float_vector(py["SHP"]) rtol=2e-6 atol=1e-12
        @test ESH ≈ _to_float_matrix(py["ESH"]) rtol=2e-6 atol=1e-12
        @test any(!iszero, fSH)
        @test all(isfinite, ESH)
    end
end

@testset "KineticH2 Alpha_H2_H fH Coupling Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KineticH2 Alpha_H2_H parity requires python with numpy/scipy"
    else
        _write_kinetic_h2_test_config()
        config_literal = JSON.json(KINETIC_H2_TEST_CONFIG)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kinetic_mesh import KineticMesh
            from KN1DPy.kinetic_h2 import KineticH2

            x = np.linspace(0.0, 0.2, 5)
            Ti = np.full(5, 2.0)
            Te = np.linspace(2.5, 4.5, 5)
            n = np.linspace(0.8e20, 1.2e20, 5)
            pipe = np.full(5, 0.1)
            mesh = KineticMesh('h2', 1, x, Ti, Te, n, pipe, E0=np.array([0.003, 0.03]), config_path=$config_literal)
            fH2BC = np.zeros((mesh.vr.size, mesh.vx.size))
            fH2BC[:, mesh.vx > 0] = 1.0
            kh2 = KineticH2(
                mesh, 1, np.zeros(mesh.x.size), fH2BC, 2.5e20,
                np.linspace(0.0, 2.0, mesh.x.size), np.zeros(mesh.x.size),
                sawada=False, compute_h_source=False, ni_correct=True,
                truncate=1.0e9, max_gen=4, config_path=$config_literal,
            )

            i = np.arange(mesh.vr.size)[:, None, None]
            j = np.arange(mesh.vx.size)[None, :, None]
            k = np.arange(mesh.x.size)[None, None, :]
            fH = 0.02 + 0.001*(i + 1) + 0.002*(j + 1) + 0.0001*(k + 1)
            kh2._compute_alpha_h_h2(fH)

            print(json.dumps({
                "fH": fH.tolist(),
                "Alpha_H2_H": kh2.Internal.Alpha_H2_H.tolist()
            }))
            """
        )

        x = collect(range(0.0, 0.2; length=5))
        mesh = KineticMesh(
            "h2",
            1,
            x,
            fill(2.0, 5),
            collect(range(2.5, 4.5; length=5)),
            collect(range(0.8e20, 1.2e20; length=5)),
            fill(0.1, 5);
            E0=[0.003, 0.03],
            config_path=KINETIC_H2_TEST_CONFIG,
        )
        fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
        @inbounds for j in eachindex(mesh.vx)
            if mesh.vx[j] > 0.0
                fH2BC[:, j] .= 1.0
            end
        end
        kh2 = KineticH2(
            mesh,
            1,
            zeros(length(mesh.x)),
            fH2BC,
            2.5e20,
            collect(range(0.0, 2.0; length=length(mesh.x))),
            zeros(length(mesh.x));
            sawada=false,
            compute_h_source=false,
            ni_correct=true,
            truncate=1.0e9,
            max_gen=4,
            initialize_static=true,
            config_path=KINETIC_H2_TEST_CONFIG,
        )

        KN1DJl._compute_alpha_h2_h!(kh2, _to_float_array3(py["fH"]))

        @test kh2.internal.Alpha_H2_H ≈ _to_float_array3(py["Alpha_H2_H"]) rtol=1e-10 atol=1e-12
        @test any(!iszero, kh2.internal.Alpha_H2_H)
    end
end

@testset "KineticH2 run_procedure Public Smoke" begin
    _write_kinetic_h2_test_config()
    x = collect(range(0.0, 0.2; length=5))
    mesh = KineticMesh(
        "h2",
        1,
        x,
        fill(2.0, 5),
        fill(3.0, 5),
        fill(1.0e20, 5),
        fill(0.1, 5);
        E0=[0.003, 0.03],
        config_path=KINETIC_H2_TEST_CONFIG,
    )
    fH2BC = zeros(Float64, length(mesh.vr), length(mesh.vx))
    @inbounds for j in eachindex(mesh.vx)
        if mesh.vx[j] > 0.0
            fH2BC[:, j] .= 1.0
        end
    end
    kh2 = KineticH2(
        mesh,
        1,
        zeros(length(mesh.x)),
        fH2BC,
        2.5e20,
        fill(1.0e20, length(mesh.x)),
        fill(1.0e15, length(mesh.x));
        sawada=false,
        compute_h_source=false,
        truncate=1.0e9,
        max_gen=4,
        config_path=KINETIC_H2_TEST_CONFIG,
    )

    result = run_procedure(
        kh2;
        fH=zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx),
        SH2=fill(1.0e15, kh2.nx),
    )

    @test result isa KH2Results
    @test size(result.fH2) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test length(result.nH2) == kh2.nx
    @test length(result.GammaxH2) == kh2.nx
    @test size(result.fSH) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(result.ESH) == (kh2.nvr, kh2.nx)
    @test all(isfinite, result.fH2)
    @test all(isfinite, result.nH2)
    @test all(result.nH2 .>= 0.0)
    @test kh2.input.fH_s !== nothing
    @test kh2.input.fH2_s === result.fH2
    @test kh2.input.nHP_s === result.nHP
    source_result = run_procedure(KineticH2(
        mesh,
        1,
        zeros(length(mesh.x)),
        fH2BC,
        2.5e20,
        fill(1.0e20, length(mesh.x)),
        fill(1.0e15, length(mesh.x));
        sawada=false,
        compute_h_source=true,
        truncate=1.0e9,
        max_gen=4,
        config_path=KINETIC_H2_TEST_CONFIG,
    ); fH=zeros(Float64, kh2.nvr, kh2.nvx, kh2.nx), SH2=fill(1.0e15, kh2.nx))
    @test size(source_result.fSH) == (kh2.nvr, kh2.nvx, kh2.nx)
    @test size(source_result.ESH) == (kh2.nvr, kh2.nx)
    @test any(!iszero, source_result.fSH)
    @test all(isfinite, source_result.ESH)
end
