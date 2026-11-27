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
    cmake --build "build/$preset" --parallel 2 --target test_surface_meshing test_mesh_tool_outputs
)
mkdir -p quad-results/ctest
ctest --test-dir "$src/build/$preset" --output-on-failure --no-tests=error -R '^test_surface_meshing "$src/build/$preset" -type f -name vc_obj2tifxyz -executable -print -quit)
test -n "$tool"
bin=$(dirname "$tool")
mkdir -p /tmp/quad-baseline
cp "$bin/vc_tifxyz2obj" "$bin/vc_obj2tifxyz" "$bin/vc_obj2tifxyz_legacy" /tmp/quad-baseline/

# Copy the baseline libraries so rebuilding the patched revision cannot\n# change the baseline.
mkdir -p /tmp/quad-baseline/lib
for tool in /tmp/quad-baseline/vc_*; do
    ldd "$bin/$(basename "$tool")" | awk '$3 ~ /^\/src\// {print $3}' | while IFS= read -r so; do
        filename=$(basename "$so")
        cp -L "$so" "/tmp/quad-baseline/lib/$filename"
        patchelf --set-rpath '$ORIGIN' "/tmp/quad-baseline/lib/$filename"
    done
    patchelf --set-rpath '$ORIGIN/lib' "$tool"
    ldd "$tool" | tee /tmp/quad-baseline/linked-libraries.txt
    if grep -q 'not found' /tmp/quad-baseline/linked-libraries.txt; then
        exit 1
    fi
done
git restore --source="$PATCHED" --worktree -- "${files[@]}"
cmake --build "$src/build/$preset" --parallel 2 --target vc_tifxyz2obj vc_obj2tifxyz vc_obj2tifxyz_legacy test_surface_meshing test_mesh_tool_outputs

# Compare decoded X/Y/Z arrays on the same public input, with matching options.
for segment in quad-results/input/*.tifxyz; do
    name=$(basename "$segment")
    python3 vesuvius-validation/verify_builds.py /tmp/quad-baseline "$bin" "$segment" "quad-results/${name:0:4}"
done

baseline_status=0
ctest --test-dir "$src/build/$preset" --output-on-failure --no-tests=error -R '^test_mesh_tool_outputs "$src/build/$preset" -type f -name vc_obj2tifxyz -executable -print -quit)
test -n "$tool"
bin=$(dirname "$tool")
mkdir -p /tmp/quad-baseline
cp "$bin/vc_tifxyz2obj" "$bin/vc_obj2tifxyz" "$bin/vc_obj2tifxyz_legacy" /tmp/quad-baseline/

# vc_core is normally a static library. Copy any local shared dependencies as
# well so rebuilding the patched revision cannot change the captured baseline.
mkdir -p /tmp/quad-baseline/lib
for tool in /tmp/quad-baseline/vc_*; do
    ldd "$bin/$(basename "$tool")" | awk '$3 ~ /^\/src\// {print $3}' | while IFS= read -r so; do
        filename=$(basename "$so")
        cp -L "$so" "/tmp/quad-baseline/lib/$filename"
        patchelf --set-rpath '$ORIGIN' "/tmp/quad-baseline/lib/$filename"
    done
    patchelf --set-rpath '$ORIGIN/lib' "$tool"
    ldd "$tool" | tee /tmp/quad-baseline/linked-libraries.txt
    if grep -q 'not found' /tmp/quad-baseline/linked-libraries.txt; then
        exit 1
    fi
done
git restore --source="$PATCHED" --worktree -- "${files[@]}"
cmake --build "$src/build/$preset" --parallel 2 --target vc_tifxyz2obj vc_obj2tifxyz vc_obj2tifxyz_legacy test_surface_meshing test_mesh_tool_outputs
ctest --test-dir "$src/build/$preset" --output-on-failure --no-tests=error -R '^(test_surface_meshing|test_mesh_tool_outputs)$'

# Compare decoded X/Y/Z arrays on the same public input, with matching options.
for segment in quad-results/input/*.tifxyz; do
    name=$(basename "$segment")
    python3 vesuvius-validation/verify_builds.py /tmp/quad-baseline "$bin" "$segment" "quad-results/${name:0:4}"
done
 > quad-results/ctest/baseline.log 2>&1 || baseline_status=$?
cat quad-results/ctest/baseline.log
tool=$(find "$src/build/$preset" -type f -name vc_obj2tifxyz -executable -print -quit)
test -n "$tool"
bin=$(dirname "$tool")
mkdir -p /tmp/quad-baseline
cp "$bin/vc_tifxyz2obj" "$bin/vc_obj2tifxyz" "$bin/vc_obj2tifxyz_legacy" /tmp/quad-baseline/

# vc_core is normally a static library. Copy any local shared dependencies as
# well so rebuilding the patched revision cannot change the captured baseline.
mkdir -p /tmp/quad-baseline/lib
for tool in /tmp/quad-baseline/vc_*; do
    ldd "$bin/$(basename "$tool")" | awk '$3 ~ /^\/src\// {print $3}' | while IFS= read -r so; do
        filename=$(basename "$so")
        cp -L "$so" "/tmp/quad-baseline/lib/$filename"
        patchelf --set-rpath '$ORIGIN' "/tmp/quad-baseline/lib/$filename"
    done
    patchelf --set-rpath '$ORIGIN/lib' "$tool"
    ldd "$tool" | tee /tmp/quad-baseline/linked-libraries.txt
    if grep -q 'not found' /tmp/quad-baseline/linked-libraries.txt; then
        exit 1
    fi
done
git restore --source="$PATCHED" --worktree -- "${files[@]}"
cmake --build "$src/build/$preset" --parallel 2 --target vc_tifxyz2obj vc_obj2tifxyz vc_obj2tifxyz_legacy test_surface_meshing test_mesh_tool_outputs
ctest --test-dir "$src/build/$preset" --output-on-failure --no-tests=error -R '^(test_surface_meshing|test_mesh_tool_outputs)$'

# Compare decoded X/Y/Z arrays on the same public input, with matching options.
for segment in quad-results/input/*.tifxyz; do
    name=$(basename "$segment")
    python3 vesuvius-validation/verify_builds.py /tmp/quad-baseline "$bin" "$segment" "quad-results/${name:0:4}"
done

ctest --test-dir "$src/build/$preset" --output-on-failure --no-tests=error -R '^test_surface_meshing$'
patched_status=0
ctest --test-dir "$src/build/$preset" --output-on-failure --no-tests=error -R '^test_mesh_tool_outputs$' > quad-results/ctest/patched.log 2>&1 || patched_status=$?
cat quad-results/ctest/patched.log

if [ "$patched_status" -ne 0 ]; then
    test "$patched_status" -eq "$baseline_status"
    for revision in baseline patched; do
        grep -E 'ERROR:|values:' "quad-results/ctest/$revision.log" > "quad-results/ctest/$revision-failures.log"
    done
    diff -u quad-results/ctest/baseline-failures.log quad-results/ctest/patched-failures.log
    echo "The mesh CLI test failed with the same assertions on the base and patch."
fi
