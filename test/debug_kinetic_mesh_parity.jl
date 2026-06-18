using JSON
using KN1DJl

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

function python_json(script::AbstractString)
    cmd = `$(PYTHON) -c $script`
    raw = read(setenv(cmd, "PYTHONPATH" => REPO_ROOT), String)
    lines = filter(!isempty, strip.(split(raw, '\n')))
    isempty(lines) && error("Python command produced no output")
    return JSON.parse(lines[end])
end

function to_float_vector(x)
    return Float64[v for v in x]
end

function print_diff_table(label::String, julia_vals::Vector{Float64}, python_vals::Vector{Float64})
    println("\n", label)
    println(rpad("idx", 6), rpad("julia", 24), rpad("python", 24), "abs_diff")
    for i in eachindex(julia_vals, python_vals)
        diff = abs(julia_vals[i] - python_vals[i])
        println(
            rpad(string(i), 6),
            rpad(string(julia_vals[i]), 24),
            rpad(string(python_vals[i]), 24),
            diff,
        )
    end
end

x = collect(range(0.0, 1.0; length=5))
Ti = fill(2.0, 5)
Te = fill(3.0, 5)
n = fill(1.0e18, 5)
PipeDia = fill(0.1, 5)

py = python_json(
    """
    import sys, json, numpy as np
    sys.path.insert(0, '.')
    from KN1DPy.kinetic_mesh import KineticMesh
    x = np.linspace(0.0, 1.0, 5)
    Ti = np.full(5, 2.0)
    Te = np.full(5, 3.0)
    n = np.full(5, 1.0e18)
    PipeDia = np.full(5, 0.1)
    km = KineticMesh('h', 1, x, Ti, Te, n, PipeDia, config_path='KN1DJl/config.json')
    print(json.dumps({
        "x": km.x.tolist(),
        "vr": km.vr.tolist(),
        "vx": km.vx.tolist(),
        "Tnorm": float(km.Tnorm)
    }))
    """
)

km = KineticMesh("h", 1, x, Ti, Te, n, PipeDia; config_path=joinpath(REPO_ROOT, "KN1DJl", "config.json"))

py_x = to_float_vector(py["x"])
py_vr = to_float_vector(py["vr"])
py_vx = to_float_vector(py["vx"])

println("KineticMesh parity debug")
println("python interpreter: ", PYTHON)
println("length(x): julia=", length(km.x), " python=", length(py_x))
println("length(vr): julia=", length(km.vr), " python=", length(py_vr))
println("length(vx): julia=", length(km.vx), " python=", length(py_vx))
println("Tnorm: julia=", km.Tnorm, " python=", Float64(py["Tnorm"]))
println("max |x diff| = ", maximum(abs.(km.x .- py_x)))

head_n = min(5, length(km.x))
tail_n = min(5, length(km.x))

print_diff_table("x head", km.x[1:head_n], py_x[1:head_n])
print_diff_table("x tail", km.x[end-tail_n+1:end], py_x[end-tail_n+1:end])
