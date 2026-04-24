python3 - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
EOF

python3 - <<'PY'
import jax
import jax.numpy as jnp
print("jax:", jax.__version__)
print("devices:", jax.devices())
print("backend:", jax.default_backend())
x = jnp.array([1., 2., 3.])
y = x + 1
print(y)
PY