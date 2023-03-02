import inspect


def run_task(task):
    prev = inspect.currentframe().f_back
    try:
        func = prev.f_globals[task]
    except KeyError:
        raise Exception(f"Task {task} not found in DAG.")
    else:
        func()
