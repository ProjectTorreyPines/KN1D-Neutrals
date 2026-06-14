using JSON
using KN1DJl

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

function interp_fixture()
    mesh_a = KineticMesh(
        "fixture",
        collect(range(0.0, 1.0; length=16)),
        fill(2.0, 16),
        fill(3.0, 16),
        fill(1.0e18, 16),
        fill(0.1, 16),
        [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5],
        [0.35, 0.7, 1.1],
        1.0,
    )
    mesh_b = KineticMesh(
        "fixture",
        collect(range(0.1, 0.9; length=12)),
        fill(2.0, 12),
        fill(3.0, 12),
        fill(1.0e18, 12),
        fill(0.1, 12),
        [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25],
        [0.4, 0.8, 1.0],
        1.0,
    )

    fa = Array{Float64,3}(undef, length(mesh_a.vr), length(mesh_a.vx), length(mesh_a.x))
    @inbounds for k in eachindex(mesh_a.x), j in eachindex(mesh_a.vx), i in eachindex(mesh_a.vr)
        fa[i, j, k] = 0.05 + 0.01 * i + 0.005 * j + 0.002 * k
    end

    return fa, mesh_a, mesh_b
end

function python_json(script::AbstractString)
    raw = read(setenv(`$(PYTHON) -c $script`, "PYTHONPATH" => REPO_ROOT), String)
    lines = filter(!isempty, strip.(split(raw, '\n')))
    isempty(lines) && error("Python command produced no output")
    return JSON.parse(lines[end])
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

function julia_time(fa, mesh_a, mesh_b; correct::Int, samples::Int=50)
    interp_fvrxx(fa, mesh_a, mesh_b; correct=correct)

    if HAS_BENCHMARKTOOLS
        return @eval BenchmarkTools.@belapsed interp_fvrxx($fa, $mesh_a, $mesh_b; correct=$correct) samples=$samples evals=1
    end

    tmin = Inf
    for _ in 1:samples
        elapsed = @elapsed interp_fvrxx(fa, mesh_a, mesh_b; correct=correct)
        tmin = min(tmin, elapsed)
    end
    return tmin
end

function julia_allocs(fa, mesh_a, mesh_b; correct::Int)
    if HAS_BENCHMARKTOOLS
        return @eval BenchmarkTools.@ballocated interp_fvrxx($fa, $mesh_a, $mesh_b; correct=$correct) evals=1
    end
    return missing
end

function python_benchmark(; correct::Int, repeat::Int=50)
    script = """
    import sys, json, time
    import numpy as np
    from types import SimpleNamespace

    sys.path.insert(0, '.')
    from KN1DPy.interp_fvrvxx import interp_fvrvxx

    mesh_a = SimpleNamespace(
        x=np.linspace(0.0, 1.0, 16),
        Ti=np.full(16, 2.0),
        Te=np.full(16, 3.0),
        ne=np.full(16, 1.0e18),
        PipeDia=np.full(16, 0.1),
        vx=np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]),
        vr=np.array([0.35, 0.7, 1.1]),
        Tnorm=1.0,
    )
    mesh_b = SimpleNamespace(
        x=np.linspace(0.1, 0.9, 12),
        Ti=np.full(12, 2.0),
        Te=np.full(12, 3.0),
        ne=np.full(12, 1.0e18),
        PipeDia=np.full(12, 0.1),
        vx=np.array([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25]),
        vr=np.array([0.4, 0.8, 1.0]),
        Tnorm=1.0,
    )
    fa = np.empty((mesh_a.vr.size, mesh_a.vx.size, mesh_a.x.size), dtype=float)
    for k in range(mesh_a.x.size):
        for j in range(mesh_a.vx.size):
            for i in range(mesh_a.vr.size):
                fa[i, j, k] = 0.05 + 0.01 * (i + 1) + 0.005 * (j + 1) + 0.002 * (k + 1)

    interp_fvrvxx(fa, mesh_a, mesh_b, correct=$correct)
    best = float("inf")
    out = None
    for _ in range($repeat):
        start = time.perf_counter()
        out = interp_fvrvxx(fa, mesh_a, mesh_b, correct=$correct)
        best = min(best, time.perf_counter() - start)

    print(json.dumps({"seconds": best, "fb": out.tolist(), "shape": list(out.shape)}))
    """
    return python_json(script)
end

function report_case(fa, mesh_a, mesh_b; correct::Int)
    fb_julia = interp_fvrxx(fa, mesh_a, mesh_b; correct=correct)
    py = python_benchmark(correct=correct)
    fb_python = to_float_array3(py["fb"])

    max_abs = maximum(abs.(fb_julia .- fb_python))
    denom = max.(abs.(fb_python), 1.0e-12)
    max_rel = maximum(abs.(fb_julia .- fb_python) ./ denom)

    jt = julia_time(fa, mesh_a, mesh_b; correct=correct)
    ja = julia_allocs(fa, mesh_a, mesh_b; correct=correct)
    pt = Float64(py["seconds"])

    println()
    println("interp_fvrxx(correct=$correct)")
    println("  shape: Julia=", size(fb_julia), " Python=", Tuple(Int.(py["shape"])))
    println("  max abs error: ", max_abs)
    println("  max rel error: ", max_rel)
    println("  Julia best:    ", round(jt * 1e6; digits=2), " us")
    println("  Python best:   ", round(pt * 1e6; digits=2), " us")
    println("  speedup:       ", round(pt / jt; digits=2), "x")
    if ja !== missing
        println("  Julia allocs:  ", ja, " bytes")
    end
end

fa, mesh_a, mesh_b = interp_fixture()

println("Threads: ", Threads.nthreads())
println("Python:  ", PYTHON)
println("BenchmarkTools: ", HAS_BENCHMARKTOOLS ? "enabled" : "not installed, using @elapsed fallback")

report_case(fa, mesh_a, mesh_b; correct=1)
report_case(fa, mesh_a, mesh_b; correct=0)
