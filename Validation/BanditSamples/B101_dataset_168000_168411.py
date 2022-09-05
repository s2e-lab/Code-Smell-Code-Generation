def batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, batch_size=None,
               feed=None,
               args=None):
  """
  A helper function that computes a tensor on numpy inputs by batches.
  This version uses exactly the tensorflow graph constructed by the
  caller, so the caller can place specific ops on specific devices
  to implement model parallelism.
  Most users probably prefer `batch_eval_multi_worker` which maps
  a single-device expression to multiple devices in order to evaluate
  faster by parallelizing across data.

  :param sess: tf Session to use
  :param tf_inputs: list of tf Placeholders to feed from the dataset
  :param tf_outputs: list of tf tensors to calculate
  :param numpy_inputs: list of numpy arrays defining the dataset
  :param batch_size: int, batch size to use for evaluation
      If not specified, this function will try to guess the batch size,
      but might get an out of memory error or run the model with an
      unsupported batch size, etc.
  :param feed: An optional dictionary that is appended to the feeding
           dictionary before the session runs. Can be used to feed
           the learning phase of a Keras model for instance.
  :param args: dict or argparse `Namespace` object.
              Deprecated and included only for backwards compatibility.
               Should contain `batch_size`
  """

  if args is not None:
    warnings.warn("`args` is deprecated and will be removed on or "
                  "after 2019-03-09. Pass `batch_size` directly.")
    if "batch_size" in args:
      assert batch_size is None
      batch_size = args["batch_size"]

  if batch_size is None:
    batch_size = DEFAULT_EXAMPLES_PER_DEVICE

  n = len(numpy_inputs)
  assert n > 0
  assert n == len(tf_inputs)
  m = numpy_inputs[0].shape[0]
  for i in range(1, n):
    assert numpy_inputs[i].shape[0] == m
  out = []
  for _ in tf_outputs:
    out.append([])
  for start in range(0, m, batch_size):
    batch = start // batch_size
    if batch % 100 == 0 and batch > 0:
      _logger.debug("Batch " + str(batch))

    # Compute batch start and end indices
    start = batch * batch_size
    end = start + batch_size
    numpy_input_batches = [numpy_input[start:end]
                           for numpy_input in numpy_inputs]
    cur_batch_size = numpy_input_batches[0].shape[0]
    assert cur_batch_size <= batch_size
    for e in numpy_input_batches:
      assert e.shape[0] == cur_batch_size

    feed_dict = dict(zip(tf_inputs, numpy_input_batches))
    if feed is not None:
      feed_dict.update(feed)
    numpy_output_batches = sess.run(tf_outputs, feed_dict=feed_dict)
    for e in numpy_output_batches:
      assert e.shape[0] == cur_batch_size, e.shape
    for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
      out_elem.append(numpy_output_batch)

  out = [np.concatenate(x, axis=0) for x in out]
  for e in out:
    assert e.shape[0] == m, e.shape
  return out