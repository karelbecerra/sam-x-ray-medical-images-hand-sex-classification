def progress_bar(iteration, total, size=30):
    running = iteration < total
    c = ">" if running else "="
    p = (size - 1) * iteration // total
    fmt = "{{:-{}d}}/{{}} [{{}}]".format(len(str(total)))
    params = [iteration, total, "=" * p + c + "." * (size - p - 1)]
    return fmt.format(*params)


def print_status_bar(iteration, total, loss, elapsed_time, epoch = None, epochs = None, metrics=None, size=30):
    metrics = " - {}  {:.4f} - ".format('loss',  loss[0])

    end = "" if iteration < total else "\n"
    if epoch == None:
      print("\r{} - {} - {:5.3f}s -{}".format('Validation', progress_bar(iteration, total), elapsed_time, metrics), end=end)
    else:
      print("\r{} - epoch: {}/{} - {} - {:5.3f}s - {}".format('Train', epoch, epochs, progress_bar(iteration, total), elapsed_time, metrics), end=end)
