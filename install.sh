#!/bin/bash

VERSIONS=(
  "0.4.23"
  "0.4.22"
  "0.4.21"
  "0.4.20"
  "0.4.18"
  "0.4.14"
)

echo "=== Start JAX auto test ==="

for VER in "${VERSIONS[@]}"; do
  echo ""
  echo "=============================="
  echo "Testing JAX version: $VER"
  echo "=============================="

  python -m pip uninstall -y jax jaxlib >/dev/null 2>&1 || true
  python -m pip install --no-cache-dir "numpy==1.26.4" >/dev/null

  if ! python -m pip install --no-cache-dir "jax[cuda12]==$VER" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; then
    echo "Install failed for $VER"
    continue
  fi

  unset LD_LIBRARY_PATH
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10

  if python - <<'PY'
import jax
import jax.numpy as jnp

print("Testing version:", jax.__version__)
print("Devices:", jax.devices())
print("Backend:", jax.default_backend())

x = jnp.arange(8)
x.block_until_ready()
print("SUCCESS")
PY
  then
    echo ""
    echo "WORKING VERSION FOUND: $VER"
    exit 0
  else
    echo "Version $VER failed, trying next one..."
  fi
done

echo ""
echo "No working version found"
exit 1