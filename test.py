import numpy as np 

def dot_product(matrix_a, matrix_b):
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

def calculU(M): 
    B = np.dot(M, M.T) 
        
    eigenvalues, eigenvectors = np.linalg.eig(B) 
    ncols = np.argsort(eigenvalues)[::-1] 
    
    return eigenvectors[:,ncols] 

def calculVt(M): 
    B = np.dot(M.T, M)
        
    eigenvalues, eigenvectors = np.linalg.eig(B) 
    ncols = np.argsort(eigenvalues)[::-1] 
    
    return eigenvectors[:,ncols].T 

def calculSigma(M): 
    mtrx1 = dot_product(M,M.T)
    mtrx2 = dot_product(M.T,M)
    if (np.size(mtrx1) > np.size(mtrx2)): 
        newM = mtrx2 
    else: 
        newM = mtrx1 
        
    eigenvalues, eigenvectors = np.linalg.eig(newM) 
    eigenvalues = np.sqrt(eigenvalues) 
    #Sorting in descending order as the svd function does 
    return eigenvalues[::-1] 

if __name__ == "__main__":
    matrix = [
                [1, 2, 3, 8],
                [4, 5, 6, 0],
                [7, 8, 9, 2],
                [7, 5, 3, 1],
                [5, 6, 0, 0] ]
    A = np.array(matrix)
    U = calculU(A) 
    Sigma = calculSigma(A) 
    Vt = calculVt(A)
    print("-------------------U-------------------")
    print(U)
    print("\n--------------Sigma----------------")
    print(Sigma)
    print("\n-------------V transpose---------------")
    print(Vt)