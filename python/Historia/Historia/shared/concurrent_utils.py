import concurrent.futures
import multiprocessing


def get_process_pool_executor(max_workers=multiprocessing.cpu_count()):
    return concurrent.futures.ProcessPoolExecutor(max_workers)


def execute_task_in_parallel(
    task_fn, inputs, max_workers=multiprocessing.cpu_count()
):
    results = {}
    with get_process_pool_executor(max_workers) as executor:
        future_results = {
            executor.submit(task_fn, *value): key
            for key, value in inputs.items()
        }
        for future in concurrent.futures.as_completed(future_results):
            try:
                results[future_results[future]] = future.result()
            except Exception:
                print("Warning: an exception was generated")
    return results
