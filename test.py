import numpy as np 
from numba import cuda, float32
import time

TPB = 16

@cuda.jit
def matmul(A, B, C):
    
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

def matrix_multiply_gpu(A,B):

    A, B = np.array(A), np.array(B)
    result_matrix = np.zeros((A.shape[0], B.shape[1]))

    blocks_size = (16, 16)
    size = (A.shape[0])*(B.shape[1])
    #print(size)
    num_blocks = (int(np.ceil(A.shape[0] / blocks_size[0])), int(np.ceil(B.shape[1] / blocks_size[0])))
    #print(num_blocks)
    matmul[num_blocks, blocks_size](A, B, result_matrix)
    #print(result_matrix.shape)
    #print(result_matrix)
    #input("?done")
    return result_matrix 

def matrix_multiply_cpu(matrix_a, matrix_b):
    # Check if the matrices have compatible dimensions
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Number of columns in Matrix A must be equal to the number of rows in Matrix B")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(matrix_b[0]))] for _ in range(len(matrix_a))]

    # Calculate the dot product
    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

def svd(M, hardware):

    multiplier = matrix_multiply_gpu if hardware=="gpu" else matrix_multiply_cpu

    def calculU(M): 
        #print(M.shape)
        B = multiplier(M, M.T) 
        
        eigenvalues, eigenvectors = np.linalg.eig(B)
        # print(eigenvalues)
        # print(eigenvectors)
        # input("?‍")
        ncols = np.argsort(eigenvalues)[::-1] 
        
        return eigenvectors[:,ncols] 

    def calculVt(M): 
        B = multiplier(M.T, M)
            
        eigenvalues, eigenvectors = np.linalg.eig(B) 
        ncols = np.argsort(eigenvalues)[::-1] 
        
        return eigenvectors[:,ncols].T 

    def calculSigma(M): 
        mtrx1 = multiplier(M,M.T)
        mtrx2 = multiplier(M.T,M)
        if (np.size(mtrx1) > np.size(mtrx2)): 
            newM = mtrx2 
        else: 
            newM = mtrx1 
            
        eigenvalues, eigenvectors = np.linalg.eig(newM) 
        eigenvalues = np.sqrt(eigenvalues) 
        #Sorting in descending order as the svd function does 
        return eigenvalues[::-1] 
    return calculU(M), calculSigma(M), calculVt(M) 

if __name__ == "__main__":
    
    # f = open("gpu_time.csv", '+a')
    # for sz in range(100,5000,100):
    #     M = np.random.randint(5, size=(sz,sz))
    #     start=time.time()
    #     U, sigma, vt = svd(M, "gpu")  
    #     f.write(f"{sz},{time.time()-start}\n")
    
    # f = open("cpu_time.csv", '+a')
    # for sz in range(100,3300,100):
    #     M = np.random.randint(5, size=(sz,sz))
    #     start=time.time()
    #     U, sigma, vt = svd(M, "cpu")
    #     f.write(f"{sz},{time.time()-start}\n") 
    
    
    M = np.array([[1,2,3,4,5], [6,7,8,9,0], [2,4,6,8,0], [1,3,5,7,9], [3,6,9,1,4]])
    U, sigma, vt = svd(M, "gpu")

    print("U: ", U)
    print("\nS:", sigma)
    print("\nVt", vt)