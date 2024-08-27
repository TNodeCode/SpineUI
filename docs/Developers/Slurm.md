# SLURM

SLURM (Simple Linux Utility for Resource Management) is an open-source workload manager that is widely used in high-performance computing (HPC) environments, such as clusters and supercomputers. It manages and schedules jobs (tasks or processes) on these systems, ensuring that resources like CPU, memory, and GPUs are allocated efficiently and fairly among users.

## Create a script

Running a script on a SLURM cluster involves several steps, including preparing your script, creating a SLURM job submission script, and submitting it to the SLURM workload manager. Here’s a step-by-step guide:

First, make sure you have the script you want to run. This could be a Python script, a shell script, or any other executable script. For example, let’s assume you have a Python script named `my_script.py`.

To run your script on a SLURM cluster, you need to create a job submission script, which is a simple shell script containing SLURM directives that specify how your job should be executed.

Here’s an example of a SLURM job submission script for `my_script.py`:

```bash
#!/bin/bash
#SBATCH --job-name=my_job         # Job name
#SBATCH --output=output_%j.txt    # Standard output and error log (%j will be replaced by job ID)
#SBATCH --ntasks=1                # Run a single task (useful for serial jobs)
#SBATCH --time=01:00:00           # Time limit hrs:min:sec (1 hour)
#SBATCH --mem=4GB                 # Memory required per node (4 GB)
#SBATCH --partition=normal        # Partition (queue) to submit to

# Load any necessary modules
module load python/3.8

# Run your Python script
python my_script.py
```

!!! note
    Ask your system administrator for the partition name you should use and the names of the available modules.

## Run a script

Once you’ve created your job submission script (e.g., submit.sh), you can submit it to the SLURM scheduler using the sbatch command:

```
$ sbatch submit.sh
```

## Inspect progress of your jobs

After submitting, SLURM will assign your job a unique job ID and place it in a queue. The job will start running when the required resources are available. You can monitor the status of your job using the `squeue` command:

```bash
$ squeue -u your_username
```

If you are interested in all jobs that run on a specific partition you can get these with the following command:

```bash
$ squeue -p partition_name
```

This can be helpful if you want to see how many capacity is left on this partition. For example if you want to train a deep learning model on GPUs you can see how many jobs are currently running that use those GPUs.


This will show the status of all your jobs. The main states are:

- PD (Pending): Waiting in the queue.
- R (Running): Currently executing.
- CG (Completing): Finishing up.
- CD (Completed): Finished successfully.

When your job completes, SLURM will write the output and any errors to the file you specified with the `--output` directive (`output_%j.txt`). You can inspect this file to see the results of your script.

## Canceling a Job

If you need to cancel a job, use the scancel command followed by the job ID:

```bash
scancel <job_id>
```