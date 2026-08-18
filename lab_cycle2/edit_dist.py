def minimum_edit_distance(source, target):
    m = len(source)
    n = len(target)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):

            if source[i - 1] == target[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    # Backtrack to find operations
    operations = []
    i = m
    j = n

    while i > 0 or j > 0:

        # Characters match
        if i > 0 and j > 0 and source[i - 1] == target[j - 1]:
            i -= 1
            j -= 1

        # Substitution
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            operations.append(
                f"Substitute '{source[i - 1]}' with '{target[j - 1]}' "
                f"at position {i}"
            )
            i -= 1
            j -= 1

        # Deletion
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            operations.append(
                f"Delete '{source[i - 1]}' at position {i}"
            )
            i -= 1

        # Insertion
        else:
            operations.append(
                f"Insert '{target[j - 1]}' at position {i + 1}"
            )
            j -= 1

    operations.reverse()

    return dp[m][n], operations


# Main program
print("MINIMUM EDIT DISTANCE")
print("-" * 30)

source = input("Enter first string: ")
target = input("Enter second string: ")

distance, operations = minimum_edit_distance(source, target)

print("\nMinimum Edit Distance:", distance)

print("\nEdit Operations:")

if operations:
    for i, operation in enumerate(operations, 1):
        print(f"{i}. {operation}")
else:
    print("No operations required. The strings are identical.")