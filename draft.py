import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn import softmax


# 定义三个向量
vector = jnp.arange(34)+1

#print(v1_expanded * v2_expanded)
# 计算三元乘积
v1=vector.reshape((34,1))
v2=vector.reshape((1,1,34))
M1=v1*vector
M2=jnp.expand_dims(M1,axis=-1)*v2
print(M1)
print(M2.shape)
#print("---------------------------------------")
upper_triangle_with_diag = jnp.triu(M2)
print(upper_triangle_with_diag.shape)