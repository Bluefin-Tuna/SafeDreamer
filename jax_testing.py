import os
import faulthandler
faulthandler.enable()

# avoid the oom errors
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.10'

# XXX: if the line below is uncommented, jax stops detecting any 
# gpu at all; but as long as this isn't set, jax sees the gpu just fine
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'gpu'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# to ensure jax finds a device:
# import jax
# print(jax.device_count())

# to ensure jax is detecting gpu backend:
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import jax.numpy as jnp

a = jnp.zeros((10,10))
b = jnp.zeros((10,10))

for _ in range(10):
    z = jnp.tensordot(a, b, 2)