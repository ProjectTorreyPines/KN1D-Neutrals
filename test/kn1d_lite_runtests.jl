using Test
using JSON
using KN1DJl

const KN1D_LITE_TEST_CONFIG = joinpath(@__DIR__, "kn1d_lite_test_config.json")

function _write_kn1d_lite_test_config()::Nothing
    open(KN1D_LITE_TEST_CONFIG, "w") do io
        JSON.print(io, Dict(
            "kinetic_h" => Dict(
                "mesh_size" => 10,
                "ion_rate" => "janev",
                "ci_test" => false,
                "alpha_cx_test" => false,
                "grid_fctr" => 0.3,
                "extra_energy_bins_eV" => Float64[],
            ),
            "kinetic_h2" => Dict(
                "mesh_size" => 6,
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
                "SIMPLE_CX" => true,
            ),
        ))
    end
    return nothing
end

@testset "KN1D lite Python Parity" begin
    if !_python_numpy_available()
        @test_skip "KN1D lite parity requires python with numpy/scipy"
    else
        _write_kn1d_lite_test_config()
        config_literal = JSON.json(KN1D_LITE_TEST_CONFIG)
        py = _python_json(
            """
            import sys, json, numpy as np
            sys.path.insert(0, '.')
            from KN1DPy.kn1d_lite import kn1d_lite

            x = np.linspace(0.0, 0.05, 5)
            r = kn1d_lite(
                x,
                1,
                np.full(5, 2.0),
                np.full(5, 3.0),
                np.full(5, 1.0e18),
                np.zeros(5),
                1.0e14,
                energies_eV=[3.0],
                fractions=[1.0],
                truncate=1.0,
                max_gen=3,
                config_path=$config_literal,
            )
            print(json.dumps({
                "shape": list(r.fH.shape),
                "Tnorm": float(r.Tnorm),
                "GammaxHBC": float(r.GammaxHBC),
                "nH": [float(r.nH[0]), float(r.nH[len(r.nH)//2]), float(r.nH[-1])],
                "GammaxH": [float(r.GammaxH[0]), float(r.GammaxH[len(r.GammaxH)//2]), float(r.GammaxH[-1])],
                "TH": [float(r.TH[0]), float(r.TH[len(r.TH)//2]), float(r.TH[-1])],
                "Sion": [float(r.Sion[0]), float(r.Sion[len(r.Sion)//2]), float(r.Sion[-1])],
                "fH_sum": float(np.sum(r.fH)),
            }))
            """
        )

        x = collect(range(0.0, 0.05; length=5))
        r = kn1d_lite(
            x,
            1.0,
            fill(2.0, 5),
            fill(3.0, 5),
            fill(1.0e18, 5),
            zeros(5),
            1.0e14;
            energies_eV=[3.0],
            fractions=[1.0],
            truncate=1.0,
            max_gen=3,
            config_path=KN1D_LITE_TEST_CONFIG,
        )
        midx = fld(length(r.nH), 2) + 1

        @test collect(size(r.fH)) == Int.(py["shape"])
        @test r.Tnorm ≈ Float64(py["Tnorm"]) atol=0 rtol=0
        @test r.GammaxHBC ≈ Float64(py["GammaxHBC"]) atol=1e12 rtol=1e-6
        @test r.nH[[1, midx, end]] ≈ _to_float_vector(py["nH"]) atol=1e8 rtol=1e-7
        @test r.GammaxH[[1, midx, end]] ≈ _to_float_vector(py["GammaxH"]) atol=1e12 rtol=1e-6
        @test r.TH[[1, midx, end]] ≈ _to_float_vector(py["TH"]) atol=1e-7 rtol=1e-6
        @test r.Sion[[1, midx, end]] ≈ _to_float_vector(py["Sion"]) atol=1e10 rtol=1e-7
        @test sum(r.fH) ≈ Float64(py["fH_sum"]) atol=1e8 rtol=1e-7
    end
end
