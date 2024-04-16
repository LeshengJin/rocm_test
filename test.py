import tvm
import numpy as np
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

# fmt: off
@T.prim_func
def fused_NT_matmul10_add10(layer_norm65: T.Buffer((1, 1, 2560), "float16"), gpt_neox_layers_0_attention_query_key_value_weight2: T.Buffer((7680, 2560), "float16"), gpt_neox_layers_0_attention_query_key_value_bias2: T.Buffer((7680,), "float16"), T_add_intermediate: T.Buffer((1, 1, 7680), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    NT_matmul_intermediate_local = T.alloc_buffer((1, 1, 7680), "float16", scope="local")
    NT_matmul_intermediate_rf_local = T.alloc_buffer((512, 1, 1, 7680), "float16", scope="local")
    NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((128, 1, 1, 7680), "float16", scope="local")
    gpt_neox_layers_0_attention_query_key_value_weight2_local = T.alloc_buffer((7680, 2560), "float16", scope="local")
    layer_norm65_shared = T.alloc_buffer((1, 1, 2560), "float16", scope="shared")
    for u_fused_ax0_fused_fused_0 in T.thread_binding(7680, thread="blockIdx.x"):
        for u_fused_ax0_fused_fused_1 in T.thread_binding(1, thread="threadIdx.y"):
            for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 in T.thread_binding(128, thread="threadIdx.x"):
                for ax0, ax1 in T.grid(1, 1):
                    for ax2_0 in T.serial(5, annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                        for ax2_1 in T.thread_binding(1, thread="threadIdx.y"):
                            for ax2_2 in T.thread_binding(128, thread="threadIdx.x"):
                                for ax2_3 in T.vectorized(4):
                                    with T.block("layer_norm65_shared"):
                                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                        v2 = T.axis.spatial(2560, ax2_0 * 512 + ax2_1 * 512 + ax2_2 * 4 + ax2_3)
                                        T.reads(layer_norm65[v0, v1, v2])
                                        T.writes(layer_norm65_shared[v0, v1, v2])
                                        layer_norm65_shared[v0, v1, v2] = layer_norm65[v0, v1, v2]
                for u_fused_ax0_fused_fused_2_init in range(1):
                    for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1_init in T.vectorized(4):
                        with T.block("NT_matmul_rf_init"):
                            vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused = T.axis.spatial(512, ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1_init)
                            v0 = T.axis.spatial(7680, u_fused_ax0_fused_fused_0 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                            T.reads()
                            T.writes(NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0])
                            NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0] = T.float16(0)
                for ax1_fused_u_fused_0 in T.serial(5, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    for ax0_ax1_fused_0 in range(4):
                        for ax0_ax1_fused_1 in T.vectorized(1):
                            with T.block("gpt_neox_layers_0_attention_query_key_value_weight2_local"):
                                v0 = T.axis.spatial(7680, u_fused_ax0_fused_fused_0)
                                v1 = T.axis.spatial(2560, ax1_fused_u_fused_0 * 512 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + ax0_ax1_fused_0 + ax0_ax1_fused_1)
                                T.reads(gpt_neox_layers_0_attention_query_key_value_weight2[v0, v1])
                                T.writes(gpt_neox_layers_0_attention_query_key_value_weight2_local[v0, v1])
                                gpt_neox_layers_0_attention_query_key_value_weight2_local[v0, v1] = gpt_neox_layers_0_attention_query_key_value_weight2[v0, v1]
                    for u_fused_ax0_fused_fused_2, ax1_fused_u_fused_2 in T.grid(1, 1):
                        for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1 in T.vectorized(4):
                            with T.block("NT_matmul_rf_update"):
                                vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused = T.axis.spatial(512, ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1)
                                v0 = T.axis.spatial(7680, u_fused_ax0_fused_fused_0 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                vax1_fused_u_fused_2, vax1_fused_u_fused_0 = T.axis.remap("RR", [ax1_fused_u_fused_2, ax1_fused_u_fused_0])
                                T.reads(NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0], layer_norm65_shared[0, 0, vax1_fused_u_fused_0 * 512 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused], gpt_neox_layers_0_attention_query_key_value_weight2_local[v0, vax1_fused_u_fused_0 * 512 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused])
                                T.writes(NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0])
                                NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0] = NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, 0, v0] + layer_norm65_shared[0, 0, vax1_fused_u_fused_0 * 512 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused] * gpt_neox_layers_0_attention_query_key_value_weight2_local[v0, vax1_fused_u_fused_0 * 512 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused]
        for ax2_fused_0_ax2_fused_1_fused in T.thread_binding(1, thread="threadIdx.y"):
            for ax0 in T.thread_binding(128, thread="threadIdx.x"):
                for ax2_fused_2_0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    for ax2_fused_2_1 in T.vectorized(1):
                        with T.block("NT_matmul_rf_init"):
                            vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v0 = T.axis.remap("SS", [ax0, u_fused_ax0_fused_fused_0])
                            T.reads()
                            T.writes(NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0])
                            NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0] = T.float16(0)
                        for ax1 in range(4):
                            with T.block("NT_matmul_rf_update"):
                                vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, v0 = T.axis.remap("SRS", [ax0, ax1, u_fused_ax0_fused_fused_0])
                                T.reads(NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0], NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, 0, 0, v0])
                                T.writes(NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0])
                                NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0] = NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0] + NT_matmul_intermediate_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, 0, 0, v0]
        for ax1_fused_2 in range(1):
            for ax1_fused_0_ax1_fused_1_fused in T.thread_binding(1, thread="threadIdx.y"):
                for ax0 in T.thread_binding(128, thread="threadIdx.x"):
                    with T.block("NT_matmul"):
                        vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, v0 = T.axis.remap("RS", [ax0, u_fused_ax0_fused_fused_0])
                        T.reads(NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0])
                        T.writes(NT_matmul_intermediate_local[0, 0, v0])
                        with T.init():
                            NT_matmul_intermediate_local[0, 0, v0] = T.float16(0)
                        NT_matmul_intermediate_local[0, 0, v0] = NT_matmul_intermediate_local[0, 0, v0] + NT_matmul_intermediate_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, 0, v0]
        for ax0_fused_0_ax0_fused_1_fused in T.thread_binding(1, thread="threadIdx.y"):
            for ax0_fused_2 in range(1):
                with T.block("T_add"):
                    v0 = T.axis.spatial(7680, u_fused_ax0_fused_fused_0)
                    T.reads(NT_matmul_intermediate_local[0, 0, v0], gpt_neox_layers_0_attention_query_key_value_bias2[v0])
                    T.writes(T_add_intermediate[0, 0, v0])
                    T_add_intermediate[0, 0, v0] = NT_matmul_intermediate_local[0, 0, v0] + gpt_neox_layers_0_attention_query_key_value_bias2[v0]
# fmt: on

mod = tvm.IRModule({"main": fused_NT_matmul10_add10})
f = tvm.build(mod["main"], target=tvm.target.Target("rocm"))
out_np = np.zeros((1, 1, 7680)).astype("float16")


def load_npz():
    input_file = "f38_fused_NT_matmul10_add10.npz"
    np_dict = np.load(input_file)
    return np_dict["arg_0"], np_dict["arg_1"], np_dict["arg_2"]


x, y, z = load_npz()


def to_tvm(tensor):
    return tvm.nd.array(tensor, device=tvm.rocm())


for i in range(10):
    tvm_out = to_tvm(out_np)
    f.entry_func(
        to_tvm(x),
        to_tvm(y),
        to_tvm(z),
        tvm_out,
    )

    # print(tvm_out)
    print(f"{i} run")
    print(f"nan num", np.sum(np.isnan(tvm_out.numpy())))
    print(f"inf num", np.sum(np.isinf(tvm_out.numpy())))
    print("=======")
