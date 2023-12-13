from numba import cuda
import numpy as np

@cuda.jit
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i,j] = tmp

if __name__ == "__main__":
    # Example matrices A and B
    matrix_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    matrix_b = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
    # print(np.matmul(matrix_a, matrix_b))
    # Ensure the result matrix has the correct shape and dtype
    result_matrix = np.empty((matrix_a.shape[0], matrix_b.shape[1]), dtype=np.float32)

    block_size = 32  # Adjust the block size based on your GPU architecture
    num_blocks_x = (result_matrix.shape[0] + block_size - 1) // block_size
    num_blocks_y = (result_matrix.shape[1] + block_size - 1) // block_size
    num_blocks = (num_blocks_x, num_blocks_y)

    # Transfer matrices to the GPU
    d_matrix_a = cuda.to_device(matrix_a)
    d_matrix_b = cuda.to_device(matrix_b)
    d_result_matrix = cuda.to_device(result_matrix)

    # Launch the kernel
    matmul[num_blocks, block_size](d_matrix_a, d_matrix_b, d_result_matrix)

    # Transfer the result back to the CPU
    d_result_matrix.copy_to_host(result_matrix)

    # Print the result
    print(result_matrix)
