import os
import subprocess
import psutil
import sys
from datetime import datetime, timezone, timedelta

def get_current_time_gmt_plus_4():
    """
    Returns the current time in GMT+4 timezone.
    """
    gmt_plus_4 = timezone(timedelta(hours=4))
    return datetime.now(gmt_plus_4).strftime("%Y-%m-%d %H:%M:%S")

def terminate_process(pid):
    """
    Terminates a specific process by its PID.
    """
    try:
        p = psutil.Process(pid)
        print(f"Terminating process {pid} ({p.name()})...")
        p.terminate()
        p.wait(5)  # Wait for process to terminate
        print(f"Process {pid} terminated successfully.")
    except psutil.NoSuchProcess:
        print(f"Process {pid} does not exist.")
    except psutil.AccessDenied:
        print(f"Access denied when trying to terminate process {pid}.")
    except Exception as e:
        print(f"Error while terminating process {pid}: {e}")

def run_python_files(file_list):
    """
    Runs the specified Python files one by one, saves their output to a file,
    and releases GPU memory by terminating the specific process after each run.
    Logs start and end times and the duration of the execution.
    """
    for python_file in file_list:
        if not os.path.isfile(python_file):
            print(f"File not found: {python_file}")
            continue

        output_file = f"{os.path.splitext(python_file)[0]}_output.log"
        start_time = get_current_time_gmt_plus_4()
        print(f"Running {python_file} at {start_time} (GMT+4)...")

        with open(output_file, "w") as f:
            # Start the process
            process = subprocess.Popen(
                ["python", python_file],
                stdout=f,
                stderr=subprocess.STDOUT
            )

            # Wait for the process to finish
            process.wait()
            end_time = get_current_time_gmt_plus_4()
            duration = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") - datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            print(f"Finished running {python_file} at {end_time} (GMT+4). Output saved to {output_file}.")
            print(f"Total execution time for {python_file}: {duration}")

            # Release GPU memory by terminating the specific process
            print(f"Releasing GPU memory used by process {process.pid}...")
            terminate_process(process.pid)
            print("GPU memory released.\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_files.py <file1.py> <file2.py> ...")
        sys.exit(1)

    files_to_run = sys.argv[1:]  # Get the list of files from command-line arguments
    run_python_files(files_to_run)
