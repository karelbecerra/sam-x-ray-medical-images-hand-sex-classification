trace_file = None

def trace(parameters, epoch, metrics_train, metrics_valid):
    global trace_file
    trace_file = trace_file if trace_file != None else open("output/trace/" + parameters.config + '_' + parameters.run + ".csv", "a")  # append mode
    row = str(epoch) \
            + ", " + str( metrics_train['accuracy'][-1]) \
            + ", " + str( metrics_valid['accuracy'][-1]) \
            + ", " + str( metrics_train['loss'][-1]) \
            + ", " + str( metrics_valid['loss'][-1]) 
    trace_file.write(row + "\n")
    trace_file.flush()


def close():
    global trace_file
    trace_file.close()
