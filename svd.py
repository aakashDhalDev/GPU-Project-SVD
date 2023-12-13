import math

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def matrix_multiply(a, b):
    result = []
    for i in range(len(a)):
        row = []
        for j in range(len(b[0])):
            total = 0
            for k in range(len(b)):
                total += a[i][k] * b[k][j]
            row.append(total)
        result.append(row)
    return result

def svd(matrix, tol=1e-10, max_iter=100):
    m, n = len(matrix), len(matrix[0])

    # Step 1: Compute A^T * A and A * A^T
    ata = matrix_multiply(transpose(matrix), matrix)
    aat = matrix_multiply(matrix, transpose(matrix))

    # Step 2: Compute eigenvalues and eigenvectors
    def power_iteration(mat, max_iter, tol):
        v = [1.0] * len(mat[0])
        for _ in range(max_iter):
            v_new = matrix_multiply(mat, [v])[0]
            norm = math.sqrt(sum(x**2 for x in v_new))
            v_new = [x / norm for x in v_new]

            # Check for convergence
            if sum((x - y)**2 for x, y in zip(v, v_new)) < tol:
                break

            v = v_new

        return v_new

    eigenvalues_ata = power_iteration(ata, max_iter, tol)
    eigenvalues_aat = power_iteration(aat, max_iter, tol)

    # Step 3: Compute singular values and vectors
    singular_values = [math.sqrt(x) for x in eigenvalues_ata]
    u = [matrix_multiply(matrix, [[v_i]])[0] for v_i in eigenvalues_aat]
    v = [matrix_multiply(transpose(matrix), [[v_i]])[0] for v_i in eigenvalues_ata]

    return u, singular_values, transpose(v)

# Example usage:
matrix = [
    [1, 2, 3, 8],
    [4, 5, 6, 0],
    [7, 8, 9, 2],
    [7, 5, 3, 1],
    [5, 6, 0, 0]
]

u, singular_values, v_t = svd(matrix)
print("U matrix:")
for row in u:
    print(row)

print("\nSingular values:", singular_values)

print("\nV^T matrix:")
for row in v_t:
    print(row)
