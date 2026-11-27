#!/usr/bin/env bash
set -euo pipefail
git config --global --add safe.directory /src
git cat-file -e "$BASE^{commit}"
export AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
bash volume-cartographer/scripts/install_build_deps.sh minimal-build
apt-get update -qq
apt-get install -y --no-install-recommends python3-pil patchelf
preset=ci-release-tests-gcc
src=volume-cartographer
git diff --name-only "$BASE" "$PATCHED" -- "$src/core" > /tmp/obj-patch-files
mapfile -t files < /tmp/obj-patch-files
test "${#files[@]}" -eq 3
git restore --source="$BASE" --worktree -- "${files[@]}"
(
    cd "$src"
    cmake --preset "$preset" -DVC_USE_CCACHE=OFF
    cmake --build "build/$preset" --parallel 2 --target vc_tifxyz2obj vc_obj2tifxyz vc_obj2tifxyz_legacy
)
tool=$(find "$src/build/$preset" -type f -name vc_obj2tifxyz -executable -print -quit)
test -n "$tool"
bin=$(dirname "$tool")
mkdir -p /tmp/quad-baseline
cp "$bin/vc_tifxyz2obj" "$bin/vc_obj2tifxyz" "$bin/vc_obj2tifxyz_legacy" /tmp/quad-baseline/

# vc_core is normally a static library. Copy any local shared dependencies as
# well so rebuilding the patched revision cannot change the captured baseline.
baseline_bin=$(realpath "$bin")
mkdir -p /tmp/quad-baseline/lib
for tool in /tmp/quad-baseline/vc_*; do
    LD_LIBRARY_PATH="$baseline_bin" ldd "$tool" | awk '$3 ~ /^\/src\// {print $3}' | while IFS= read -r so; do
        filename=$(basename "$so")
        cp -L "$so" "/tmp/quad-baseline/lib/$filename"
        patchelf --set-rpath '$ORIGIN' "/tmp/quad-baseline/lib/$filename"
    done
    patchelf --set-rpath '$ORIGIN/lib' "$tool"
done
git restore --source="$PATCHED" --worktree -- "${files[@]}"
cmake --build "$src/build/$preset" --parallel 2 --target vc_tifxyz2obj vc_obj2tifxyz vc_obj2tifxyz_legacy test_surface_meshing test_mesh_tool_outputs
ctest --test-dir "$src/build/$preset" --output-on-failure --no-tests=error -R '^(test_surface_meshing|test_mesh_tool_outputs)$'

# Compare decoded X/Y/Z arrays on the same public input, with matching options.
for segment in quad-results/input/*.tifxyz; do
    name=$(basename "$segment")
    python3 vesuvius-validation/verify_builds.py /tmp/quad-baseline "$bin" "$segment" "quad-results/${name:0:4}"
done
