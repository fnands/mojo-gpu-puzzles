from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


alias SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
alias layout = Layout.row_major(SIZE)
alias GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
alias BLOCK_DIM_X = 1 << log2_ceil(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[mut=True, dtype, layout],
    input: LayoutTensor[mut=False, dtype, layout],
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    shared_max = tb[dtype]().row_major[SIZE]().shared().alloc()
    shared_sum = tb[dtype]().row_major[SIZE]().shared().alloc()
    shared_input = tb[dtype]().row_major[SIZE]().shared().alloc()
    shared_exponents = tb[dtype]().row_major[SIZE]().shared().alloc()

    
    if global_i < SIZE:
        shared_input[local_i] = input[global_i][0]
        shared_max[local_i] = shared_input[local_i]
    

    barrier()

    var stride = SIZE // 2
    # Get max
    @parameter
    for _ in range(BLOCK_DIM_X):
        if local_i < stride:
            if shared_max[local_i] < shared_max[local_i + stride]:
                shared_max[local_i] = shared_max[local_i + stride]
        stride //= 2
        barrier()
        



    # Get exponents
    if global_i < SIZE:
        shared_exponents[local_i] = exp(shared_input[local_i] - shared_max[0])
        shared_sum[local_i] = shared_exponents[local_i]

    
    barrier()

    stride = SIZE // 2
    # Get sum
    @parameter
    for _ in range(BLOCK_DIM_X):
        if local_i < stride:
            shared_sum[local_i] += shared_sum[local_i + stride]

        stride //= 2      
        barrier()
        

    #print(shared_sum[0])

    if global_i < SIZE:
        output[global_i] = shared_exponents[local_i] / shared_sum[0]

        #print(output[local_i])



    


# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutableAnyOrigin],
    input: LayoutTensor[dtype, layout, MutableAnyOrigin],
):
    
    # smallest possible value expressed by dtype
    var max_val: Scalar[dtype] = min_finite[dtype]()
    var sum_exponents: input.element_type = 0
    


    @parameter
    for i in range(input_size):
        max_val = max(rebind[Scalar[dtype]](input[i]), max_val)

    @parameter
    for i in range(input_size):
        output[i] = exp(rebind[Scalar[dtype]](input[i]) - max_val)
        sum_exponents += output[i]

    

    @parameter
    for i in range(input_size):
        output[i] = output[i] / sum_exponents
    



# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](output.to_layout_tensor())
        var input_tensor = rebind[
            LayoutTensor[dtype, layout, MutableAnyOrigin]
        ](input.to_layout_tensor())

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[UnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            gpu_ctx.enqueue_function[
                softmax_gpu_kernel[layout, input_size, dtype]
            ](
                output_tensor,
                input_tensor,
                grid_dim=GRID_DIM_X,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
