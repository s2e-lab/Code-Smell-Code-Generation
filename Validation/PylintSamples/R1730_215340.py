def _damerau_levenshtein(a, b):
  """Returns Damerau-Levenshtein edit distance from a to b."""
  memo = {}

  def distance(x, y):
    """Recursively defined string distance with memoization."""
    if (x, y) in memo:
      return memo[x, y]
    if not x:
      d = len(y)
    elif not y:
      d = len(x)
    else:
      d = min(
          distance(x[1:], y) + 1,  # correct an insertion error
          distance(x, y[1:]) + 1,  # correct a deletion error
          distance(x[1:], y[1:]) + (x[0] != y[0]))  # correct a wrong character
      if len(x) >= 2 and len(y) >= 2 and x[0] == y[1] and x[1] == y[0]:
        # Correct a transposition.
        t = distance(x[2:], y[2:]) + 1
        if d > t:
          d = t

    memo[x, y] = d
    return d
  return distance(a, b)