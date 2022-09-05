def cardinality(gym_space):
  """Number of elements that can be represented by the space.

  Makes the most sense for Discrete or Box type with integral dtype, ex: number
  of actions in an action space.

  Args:
    gym_space: The gym space.

  Returns:
    np.int64 number of observations that can be represented by this space, or
    returns None when this doesn't make sense, i.e. float boxes etc.

  Raises:
    NotImplementedError when a space's cardinality makes sense but we haven't
    implemented it.
  """

  if (gym_space.dtype == np.float32) or (gym_space.dtype == np.float64):
    tf.logging.error("Returning None for a float gym space's cardinality: ",
                     gym_space)
    return None

  if isinstance(gym_space, Discrete):
    return gym_space.n

  if isinstance(gym_space, Box):
    # Construct a box with all possible values in this box and take a product.
    return np.prod(gym_space.high - gym_space.low + 1)

  raise NotImplementedError