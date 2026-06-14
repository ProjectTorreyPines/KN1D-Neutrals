using Test
using NPZ
using KN1DJl

const DIIID_DIR = normpath(joinpath(@__DIR__, "..", "..", "examples", "DIII-D"))
const DIIID_INPUT_NPZ = joinpath(DIIID_DIR, "python_output", "KN1D_input.npz")
const DIIID_CONFIG = joinpath(DIIID_DIR, "python_output", "config.json")

function _python_scipy_available()::Bool
    try
        run(pipeline(setenv(`$(PYTHON) -c "import numpy, scipy"`, "PYTHONPATH" => REPO_ROOT); stdout=devnull, stderr=devnull))
        return true
    catch
        return false
    end
end

function _diiid_h_fctr(gauge_h2::Float64, config_path::AbstractString)::Union{Nothing,Float64}
    gauge_h2 > 30.0 || return nothing
    return get_config(String(config_path)).kinetic_h.grid_fctr * 30.0 / gauge_h2
end

@testset "DIII-D KineticMesh Python Fixture Parity" begin
    if !isfile(DIIID_INPUT_NPZ)
        @test_skip "DIII-D python_output fixture is not available"
    else
        inp = npzread(DIIID_INPUT_NPZ)
        fctr = _diiid_h_fctr(Float64(inp["GaugeH2"]), DIIID_CONFIG)

        mesh = KineticMesh(
            "h",
            Int(inp["mu"]),
            Vector{Float64}(inp["x"]),
            Vector{Float64}(inp["Ti"]),
            Vector{Float64}(inp["Te"]),
            Vector{Float64}(inp["n"]),
            Vector{Float64}(inp["PipeDia"]);
            jh=default_johnson_hinnov(),
            E0=[0.0],
            fctr=fctr,
            config_path=DIIID_CONFIG,
        )

        @test length(mesh.x) == length(inp["xH"])
        @test mesh.vr ≈ Vector{Float64}(inp["vrA"]) atol=0 rtol=0
        @test mesh.vx ≈ Vector{Float64}(inp["vxA"]) atol=0 rtol=0
        # The Python fixture was saved from a prior run; tiny x/Tnorm drift is
        # expected from interpolation and rate-evaluation implementation details.
        @test mesh.x ≈ Vector{Float64}(inp["xH"]) atol=1e-6 rtol=1e-8
        @test mesh.Tnorm ≈ Float64(inp["TnormA"]) atol=1e-3 rtol=1e-6
    end
end

@testset "DIII-D Profile KN1D-lite Python Parity" begin
    if !isfile(DIIID_INPUT_NPZ)
        @test_skip "DIII-D python_output fixture is not available"
    elseif !_python_scipy_available()
        @test_skip "DIII-D lite parity requires python with numpy/scipy"
    else
        inp = npzread(DIIID_INPUT_NPZ)
        incident_n0 = 1.0e14
        max_gen = 50
        truncate = 1.0e-3
        config_literal = JSON.json(DIIID_CONFIG)
        input_literal = JSON.json(DIIID_INPUT_NPZ)

        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kn1d_lite import kn1d_lite

            inp = np.load($input_literal)
            r = kn1d_lite(
                inp["x"],
                int(inp["mu"]),
                inp["Ti"],
                inp["Te"],
                inp["n"],
                inp["vxi"],
                $incident_n0,
                energies_eV=[3.0],
                fractions=[1.0],
                truncate=$truncate,
                max_gen=$max_gen,
                config_path=$config_literal,
            )
            print(json.dumps({
                "shape": list(r.fH.shape),
                "Tnorm": float(r.Tnorm),
                "xH": r.xH.tolist(),
                "vr": r.vr.tolist(),
                "vx": r.vx.tolist(),
                "fH_sum": float(np.sum(r.fH)),
                "nH": r.nH.tolist(),
                "GammaxH": r.GammaxH.tolist(),
                "TH": r.TH.tolist(),
                "Sion": r.Sion.tolist(),
            }))
            """
        )

        r = kn1d_lite(
            Vector{Float64}(inp["x"]),
            Int(inp["mu"]),
            Vector{Float64}(inp["Ti"]),
            Vector{Float64}(inp["Te"]),
            Vector{Float64}(inp["n"]),
            Vector{Float64}(inp["vxi"]),
            incident_n0;
            energies_eV=[3.0],
            fractions=[1.0],
            truncate=truncate,
            max_gen=max_gen,
            config_path=DIIID_CONFIG,
        )

        @test collect(size(r.fH)) == Int.(py["shape"])
        # This realistic profile exercises adaptive mesh generation and
        # Johnson-Hinnov spline rates. Tiny mesh/Tnorm differences produce
        # ppm-level downstream differences, so these tolerances are looser than
        # the synthetic unit fixtures while still catching convention mistakes.
        @test r.Tnorm ≈ Float64(py["Tnorm"]) atol=1e-3 rtol=1e-6
        @test r.xH ≈ _to_float_vector(py["xH"]) atol=1e-6 rtol=1e-6
        @test r.vr ≈ _to_float_vector(py["vr"]) atol=1e-8 rtol=1e-8
        @test r.vx ≈ _to_float_vector(py["vx"]) atol=1e-8 rtol=1e-8
        @test sum(r.fH) ≈ Float64(py["fH_sum"]) atol=1e8 rtol=1e-5
        @test r.nH ≈ _to_float_vector(py["nH"]) atol=1e8 rtol=1e-5
        @test r.GammaxH ≈ _to_float_vector(py["GammaxH"]) atol=1e12 rtol=1e-5
        @test r.TH ≈ _to_float_vector(py["TH"]) atol=1e-6 rtol=1e-6
        @test r.Sion ≈ _to_float_vector(py["Sion"]) atol=1e10 rtol=1e-5
    end
end
