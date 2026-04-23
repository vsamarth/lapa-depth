

# pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt


# pip ubinstall jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# # pip install --upgrade "jax[cuda12]"
# pip uninstall  "jax[cuda12]"


# pip install jax==0.4.23
# pip install jaxlib==0.4.23 \
#   -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python - <<'PY'
import jax
import jax.numpy as jnp
print("jax:", jax.__version__)
print("devices:", jax.devices())
print("backend:", jax.default_backend())
x = jnp.array([1., 2., 3.])
y = x + 1
print(y)
PY