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

if __name__=="__main__":
    # Example matrices A and B
    matrix_a = [[1, 2, 3], [4, 5, 6]]
    matrix_b = [[7, 8], [9, 10], [11, 12]]

    # Calculate the dot product
    result_matrix = dot_product(matrix_a, matrix_b)

    # Print the result
    for row in result_matrix:
        print(row)
