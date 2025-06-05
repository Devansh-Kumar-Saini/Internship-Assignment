def lower_triangular(n):
    for i in range(1, n + 1):
        print('*' * i)

def upper_triangular(n):
    for i in range(n, 0, -1):
        print('*' * i)

def pyramid(n):
    for i in range(n):
        print(' ' * (n - i - 1) + '*' * (2 * i + 1))

# Set the number of rows
n = 5

print("Lower Triangular:")
lower_triangular(n)
print("\nUpper Triangular:")
upper_triangular(n)
print("\nPyramid:")
pyramid(n)