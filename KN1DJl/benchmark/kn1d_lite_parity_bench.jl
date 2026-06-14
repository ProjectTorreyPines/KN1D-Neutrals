using JSON
using KN1DJl
using Plots

const HAS_BENCHMARKTOOLS = let
    try
        @eval using BenchmarkTools
        true
    catch
        false
    end
end

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const PYTHON = let
    override = get(ENV, "KN1DJL_PYTHON", "")
    if !isempty(override)
        override
    else
        venv_python = joinpath(REPO_ROOT, ".venv", "bin", "python")
        isfile(venv_python) ? venv_python : "python3"
    end
end

const PARITY_CONFIG = joinpath(@__DIR__, "kn1d_lite_parity_config.json")

function write_parity_config()::Nothing
    open(PARITY_CONFIG, "w") do io
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

function python_json(script::AbstractString)
    raw = read(setenv(`$(PYTHON) -c $script`, "PYTHONPATH" => REPO_ROOT), String)
    lines = filter(!isempty, strip.(split(raw, '\n')))
    isempty(lines) && error("Python command produced no output")
    return JSON.parse(lines[end])
end

function to_float_vector(x)
    return Float64[v for v in x]
end

function to_float_array3(x)
    n1 = length(x)
    n2 = n1 == 0 ? 0 : length(x[1])
    n3 = (n1 == 0 || n2 == 0) ? 0 : length(x[1][1])
    A = Array{Float64,3}(undef, n1, n2, n3)
    @inbounds for i in 1:n1, j in 1:n2, k in 1:n3
        A[i, j, k] = Float64(x[i][j][k])
    end
    return A
end

function fixture()
    return (
        x=collect(range(0.0, 0.05; length=5)),
        mu=1,
        Ti=fill(2.0, 5),
        Te=fill(3.0, 5),
        n=fill(1.0e18, 5),
        vxi=zeros(Float64, 5),
        incident_n0=1.0e14,
        energies_eV=[3.0],
        fractions=[1.0],
        truncate=1.0,
        max_gen=3,
        config_path=PARITY_CONFIG,
    )
end

function run_julia_case()
    f = fixture()
    return kn1d_lite(
        f.x,
        f.mu,
        f.Ti,
        f.Te,
        f.n,
        f.vxi,
        f.incident_n0;
        energies_eV=f.energies_eV,
        fractions=f.fractions,
        truncate=f.truncate,
        max_gen=f.max_gen,
        config_path=f.config_path,
    )
end

function prepare_julia_case()
    f = fixture()
    return prepare_kn1d_lite(
        f.x,
        f.mu,
        f.Ti,
        f.Te,
        f.n,
        f.vxi,
        f.incident_n0;
        energies_eV=f.energies_eV,
        fractions=f.fractions,
        truncate=f.truncate,
        max_gen=f.max_gen,
        config_path=f.config_path,
    )
end

function best_elapsed(f::Function; samples::Int=10)::Float64
    f()
    if HAS_BENCHMARKTOOLS
        return @eval BenchmarkTools.@belapsed ($f)() samples=$samples evals=1
    end

    best = Inf
    for _ in 1:samples
        best = min(best, @elapsed f())
    end
    return best
end

function julia_cold_time(; samples::Int=10)
    return best_elapsed(run_julia_case; samples=samples)
end

function julia_warm_time(problem::KN1DLiteProblem; samples::Int=50)
    return best_elapsed(() -> run_kn1d_lite(problem); samples=samples)
end

function build_boundary_condition(mesh::KineticMesh, f)::Tuple{Matrix{Float64},Float64}
    component_vs, component_fractions, _ = KN1DJl._component_speeds(f.mu, f.energies_eV, nothing, f.fractions)
    vth = sqrt(2.0 * KN1DJl.Q * mesh.Tnorm / (f.mu * KN1DJl.H_MASS))
    vdiff = VSpaceDifferentials(mesh.vr, mesh.vx)
    fHBC = zeros(Float64, length(mesh.vr), length(mesh.vx))
    GammaxHBC = 0.0

    @inbounds for m in eachindex(component_vs, component_fractions)
        v_ms = component_vs[m]
        v_norm = v_ms / vth
        ix = argmin(abs.(mesh.vx .- v_norm))
        fHBC[1, ix] += (component_fractions[m] * f.incident_n0) /
                        (vdiff.dvr_vol[1] * vdiff.dvx[ix])
        GammaxHBC += component_fractions[m] * f.incident_n0 * v_ms
    end

    return fHBC, GammaxHBC
end

function construct_mesh(f)::KineticMesh
    _, _, E0 = KN1DJl._component_speeds(f.mu, f.energies_eV, nothing, f.fractions)
    return KineticMesh(
        "h",
        f.mu,
        f.x,
        f.Ti,
        f.Te,
        f.n,
        zeros(Float64, length(f.x));
        E0=E0,
        config_path=f.config_path,
    )
end

function construct_static_kinetic_h(mesh::KineticMesh, fHBC::Matrix{Float64}, GammaxHBC::Float64, f)::KineticH
    vxiA = interp_1d(f.x, f.vxi, mesh.x; fill_value="extrapolate")
    return KineticH(
        mesh,
        f.mu,
        vxiA,
        fHBC,
        GammaxHBC;
        ni_correct=true,
        truncate=Float64(f.truncate),
        max_gen=Int(f.max_gen),
        compute_errors=false,
        debrief=0,
        debug=0,
        config_path=f.config_path,
        initialize_static=true,
    )
end

function section_times(; samples::Int=10)
    f = fixture()
    mesh = construct_mesh(f)
    fHBC, GammaxHBC = build_boundary_condition(mesh, f)
    problem = prepare_julia_case()

    mesh_time = best_elapsed(() -> construct_mesh(f); samples=samples)
    bc_time = best_elapsed(() -> build_boundary_condition(mesh, f); samples=samples)
    static_kh_time = best_elapsed(() -> construct_static_kinetic_h(mesh, fHBC, GammaxHBC, f); samples=samples)
    prepare_time = best_elapsed(prepare_julia_case; samples=samples)
    warm_time = julia_warm_time(problem; samples=max(3 * samples, 30))

    return (
        cold_names=["Kinetic mesh", "Boundary condition", "Static KineticH", "Full prepare"],
        cold_seconds=[mesh_time, bc_time, static_kh_time, prepare_time],
        runtime_names=["Julia cold", "Julia warm", "Python"],
        warm_seconds=warm_time,
    )
end

function run_python_case(; repeat::Int=10)
    config_path = JSON.json(PARITY_CONFIG)
    script = """
    import sys, json, time, io, contextlib
    import numpy as np
    from scipy import interpolate
    sys.path.insert(0, '.')
    from KN1DPy.kn1d_lite import kn1d_lite
    from KN1DPy.kinetic_mesh import KineticMesh
    from KN1DPy.kinetic_h import KineticH
    from KN1DPy.make_dvr_dvx import VSpace_Differentials
    from KN1DPy.rates.johnson_hinnov.johnson_hinnov import Johnson_Hinnov
    from KN1DPy.common import constants as CONST

    config_path = $config_path
    x = np.linspace(0.0, 0.05, 5)
    Ti = np.full(5, 2.0)
    Te = np.full(5, 3.0)
    n = np.full(5, 1.0e18)
    vxi = np.zeros(5)
    mu = 1
    incident_n0 = 1.0e14
    energies_eV = np.array([3.0])
    fractions = np.array([1.0])
    truncate = 1.0
    max_gen = 3
    jh = Johnson_Hinnov()

    def quiet(fn):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn()

    def best(fn, repeats=$repeat):
        fn()
        best_time = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            best_time = min(best_time, time.perf_counter() - start)
        return best_time

    def build_mesh():
        return quiet(lambda: KineticMesh(
            'h', mu, x, Ti, Te, n, np.zeros_like(x),
            jh=jh,
            E0=energies_eV,
            config_path=config_path,
        ))

    def build_boundary_condition(kh_mesh):
        component_vs = np.sqrt(2.0 * CONST.Q * energies_eV / (mu * CONST.H_MASS))
        vth = np.sqrt(2.0 * CONST.Q * kh_mesh.Tnorm / (mu * CONST.H_MASS))
        kh_differentials = VSpace_Differentials(kh_mesh.vr, kh_mesh.vx)
        fHBC = np.zeros((kh_mesh.vr.size, kh_mesh.vx.size))
        GammaxHBC = 0.0
        for frac, v_ms in zip(fractions, component_vs):
            v_norm = v_ms / vth
            ix = int(np.argmin(np.abs(kh_mesh.vx - v_norm)))
            fHBC[0, ix] += (frac * incident_n0) / (kh_differentials.dvr_vol[0] * kh_differentials.dvx[ix])
            GammaxHBC += frac * incident_n0 * v_ms
        return fHBC, GammaxHBC

    def build_static_kinetic_h(kh_mesh, fHBC, GammaxHBC):
        vxiA = interpolate.interp1d(x, vxi, fill_value='extrapolate')(kh_mesh.x)
        return KineticH(
            kh_mesh, mu, vxiA, fHBC, GammaxHBC,
            jh=jh,
            ni_correct=True,
            truncate=truncate,
            max_gen=max_gen,
            compute_errors=False,
            debrief=False,
            debug=False,
            config_path=config_path,
        )

    def prepare_only():
        kh_mesh = build_mesh()
        fHBC, GammaxHBC = build_boundary_condition(kh_mesh)
        kinetic_h = build_static_kinetic_h(kh_mesh, fHBC, GammaxHBC)
        return kh_mesh, fHBC, GammaxHBC, kinetic_h

    def run():
        return kn1d_lite(
            x, mu, Ti, Te, n, vxi, incident_n0,
            energies_eV=energies_eV,
            fractions=fractions,
            truncate=truncate,
            max_gen=max_gen,
            config_path=config_path,
        )

    r = run()
    section_mesh = build_mesh()
    section_fHBC, section_GammaxHBC = build_boundary_condition(section_mesh)

    best_run = best(run)
    mesh_seconds = best(build_mesh)
    bc_seconds = best(lambda: build_boundary_condition(section_mesh))
    static_seconds = best(lambda: build_static_kinetic_h(section_mesh, section_fHBC, section_GammaxHBC))
    prepare_seconds = best(prepare_only)

    r = run()
    print(json.dumps({
        "seconds": best_run,
        "section_seconds": {
            "Kinetic mesh": mesh_seconds,
            "Boundary condition": bc_seconds,
            "Static KineticH": static_seconds,
            "Full prepare": prepare_seconds
        },
        "xH": r.xH.tolist(),
        "vr": r.vr.tolist(),
        "vx": r.vx.tolist(),
        "Tnorm": float(r.Tnorm),
        "fH": r.fH.tolist(),
        "nH": r.nH.tolist(),
        "GammaxH": r.GammaxH.tolist(),
        "VxH": r.VxH.tolist(),
        "TH": r.TH.tolist(),
        "qxH_total": r.qxH_total.tolist(),
        "Sion": r.Sion.tolist(),
        "fHBC": r.fHBC.tolist(),
        "GammaxHBC": float(r.GammaxHBC),
    }))
    """
    return python_json(script)
end

function max_errors(j, py)
    fields = (
        :fH => (j.fH, to_float_array3(py["fH"])),
        :nH => (j.nH, to_float_vector(py["nH"])),
        :GammaxH => (j.GammaxH, to_float_vector(py["GammaxH"])),
        :VxH => (j.VxH, to_float_vector(py["VxH"])),
        :TH => (j.TH, to_float_vector(py["TH"])),
        :qxH_total => (j.qxH_total, to_float_vector(py["qxH_total"])),
        :Sion => (j.Sion, to_float_vector(py["Sion"])),
    )

    names = String[]
    abs_errors = Float64[]
    rel_errors = Float64[]
    for (name, (a, b)) in fields
        push!(names, String(name))
        diff = abs.(a .- b)
        push!(abs_errors, isempty(diff) ? 0.0 : maximum(diff))
        denom = max.(abs.(b), 1.0e-30)
        push!(rel_errors, isempty(diff) ? 0.0 : maximum(diff ./ denom))
    end
    return names, abs_errors, rel_errors
end

function annotate_bars!(plt, xs, ys; digits::Int=2, suffix::String=" ms")
    for (x, y) in zip(xs, ys)
        annotate!(plt, x, y * 1.12, text(string(round(y; digits=digits), suffix), 8, :center))
    end
    return plt
end

function make_plot(names, abs_errors, rel_errors, jtc, jtw, pt, sections, py_sections)
    theme(:wong2)
    default(
        fontfamily="Computer Modern",
        framestyle=:box,
        grid=true,
        gridalpha=0.25,
        foreground_color_axis=:gray25,
        foreground_color_grid=:gray80,
        titlefontsize=12,
        guidefontsize=10,
        tickfontsize=8,
        legendfontsize=9,
        dpi=220,
    )

    runtime_ms = [jtc, jtw, pt] .* 1.0e3
    speed_plot = bar(
        sections.runtime_names,
        runtime_ms;
        yaxis=:log10,
        ylabel="best time (ms, log scale)",
        title="End-to-End Runtime",
        legend=false,
        color=[:steelblue :seagreen :darkorange],
        margin=6Plots.mm,
    )
    annotate_bars!(speed_plot, 1:length(runtime_ms), runtime_ms)

    cold_ms = sections.cold_seconds .* 1.0e3
    cold_plot = bar(
        sections.cold_names,
        cold_ms;
        yaxis=:log10,
        ylabel="best time (ms, log scale)",
        title="Cold-Path Section Timings",
        legend=false,
        color=:steelblue,
        xrotation=25,
        margin=6Plots.mm,
    )
    annotate_bars!(cold_plot, 1:length(cold_ms), cold_ms)

    py_cold_ms = [Float64(py_sections[name]) for name in sections.cold_names] .* 1.0e3
    py_section_plot = bar(
        sections.cold_names,
        py_cold_ms;
        yaxis=:log10,
        ylabel="best time (ms, log scale)",
        title="Python Cold-Path Section Timings",
        legend=false,
        color=:darkorange,
        xrotation=25,
        margin=6Plots.mm,
    )
    annotate_bars!(py_section_plot, 1:length(py_cold_ms), py_cold_ms)

    err_plot = bar(
        names,
        rel_errors .+ eps();
        yaxis=:log10,
        ylabel="max relative error",
        title="Julia vs Python Parity",
        legend=false,
        color=:purple,
        xrotation=35,
        margin=6Plots.mm,
    )

    fig = plot(
        speed_plot,
        cold_plot,
        py_section_plot,
        err_plot;
        layout=(2, 2),
        size=(1450, 900),
        plot_title="KN1D-lite Julia/Python Benchmark and Section Breakdown",
        plot_titlefontsize=15,
    )
    png_out = joinpath(@__DIR__, "kn1d_lite_parity.png")
    svg_out = joinpath(@__DIR__, "kn1d_lite_parity.svg")
    savefig(fig, png_out)
    savefig(fig, svg_out)
    return png_out, svg_out
end

println("Threads: ", Threads.nthreads())
println("Python:  ", PYTHON)
println("BenchmarkTools: ", HAS_BENCHMARKTOOLS ? "enabled" : "not installed, using @elapsed fallback")

write_parity_config()
j = run_julia_case()
py = run_python_case()
problem = prepare_julia_case()
jtc = julia_cold_time()
jtw = julia_warm_time(problem)
pt = Float64(py["seconds"])
sections = section_times()
names, abs_errors, rel_errors = max_errors(j, py)
png_path, svg_path = make_plot(names, abs_errors, rel_errors, jtc, jtw, pt, sections, py["section_seconds"])

println()
println("KN1D lite parity")
println("  shape: Julia fH=", size(j.fH), " Python fH=", size(to_float_array3(py["fH"])))
println("  Tnorm: Julia=", j.Tnorm, " Python=", Float64(py["Tnorm"]))
println("  GammaxHBC: Julia=", j.GammaxHBC, " Python=", Float64(py["GammaxHBC"]))
for i in eachindex(names)
    println("  ", rpad(names[i], 11), " max_abs=", abs_errors[i], " max_rel=", rel_errors[i])
end
println("  Julia cold best: ", round(jtc * 1e3; digits=3), " ms")
println("  Julia warm best: ", round(jtw * 1e3; digits=3), " ms")
println("  Python best:     ", round(pt * 1e3; digits=3), " ms")
println("  cold speedup:    ", round(pt / jtc; digits=2), "x")
println("  warm speedup:    ", round(pt / jtw; digits=2), "x")
println()
println("Section timings")
println("  ", rpad("section", 20), lpad("Julia", 12), lpad("Python", 12))
for i in eachindex(sections.cold_names)
    name = sections.cold_names[i]
    println(
        "  ",
        rpad(name, 20),
        lpad(string(round(sections.cold_seconds[i] * 1e3; digits=3)), 10), " ms",
        lpad(string(round(Float64(py["section_seconds"][name]) * 1e3; digits=3)), 10), " ms",
    )
end
println("  ", rpad("Warm solve", 20), lpad(string(round(sections.warm_seconds * 1e3; digits=3)), 10), " ms", lpad("n/a", 13))
println()
println("Plots")
println("  png: ", png_path)
println("  svg: ", svg_path)
