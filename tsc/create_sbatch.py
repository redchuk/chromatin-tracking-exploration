import click
from datetime import datetime
from jinja2 import Environment
from pathlib import Path, PosixPath
import re
import subprocess

BASH = """#!/bin/bash
#

#SBATCH --job-name={{ job_name }}
#SBATCH -M {{ cluster }}
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000
#SBATCH -p {{ partition }}
#SBATCH --gres=gpu:1

# Time limit
#SBATCH -t {{ time }}

#SBATCH --chdir={{ job_dir }}
#SBATCH --output={{ job_name }}-%j.out


# Load the Conda module
module use /proj/group/lmu/envs_and_modules/slurm_modules/
module --ignore-cache load Miniconda3/4.12.0_py38
source activate tsc
conda env list
which python

echo "Starting..."
date

PROG="{{ prog_dir }}/{{ prog }}"
PATHS="{{ paths }}"
OPTIONS="{{ options }} --job_name={{ job_name }} --job_id=$SLURM_JOBID"

cmd="srun python ${PROG} --paths ${PATHS} ${OPTIONS}"
echo ${cmd}
${cmd}

echo "Done."
date
"""

def git_clone(repo, branch):
    tmpstr = "tmp_clone_" + datetime.now().strftime("%H%M%S%f")
    tmp = PosixPath(tmpstr)
    
    cmd = "git clone -b %s --single-branch --depth 1 %s %s" % (branch, repo, str(tmp))
    subprocess.run(cmd.split())

    # save only the scripts directory
    for subdir in ['data', 'notebooks']:
        if (tmp / subdir).is_dir():
            for f in (tmp / subdir).iterdir():
                f.unlink()
            (tmp / subdir).rmdir()

    cmd = "git --git-dir ./" + tmpstr + "/.git log -n 1 --pretty=format:'%H'"
    result = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
    commit = result.stdout.decode().replace("'","")
    print(commit)

    root = Path("/proj/hajaalin/Projects/n_track_ML/run")
    project = re.search(".*/(.*).git$", repo).groups()[0]
    prog_dir = root / (project + "_" + commit)
    if not prog_dir.exists():
        tmp.rename(prog_dir)
    #else:
    #    cmd = "rm -rf %s" % str(Path(__file__).parent.absolute() / tmpstr)
    #    subprocess.run(cmd)

    return prog_dir

@click.command()
@click.option("--job_name", type=str, default="tsc-it")
@click.option("--job_dir", type=str, default="/wrk-vakka/users/hajaalin/output/TSC")
@click.option("--cluster", type=click.Choice(['ukko','kale']), default="ukko")
@click.option("--partition", type=str, default="gpu,gpu-oversub")
@click.option("--time", type=str, default="4:00:00")
@click.option("--test", is_flag=True, default=False, show_default=True)
@click.option("--prog", type=str, default="cv_inceptiontime.py")
@click.option("--branch_n", type=str)
@click.option("--branch_i", type=str)
@click.option("--paths", type=str, default="paths.yml")
@click.option("--options", type=str, default="'--epochs=100 --kernel_size=15 --repeats=20'")
@click.option("--sbatch_dir", type=str, default="./sbatch")
@click.option("--loop_epochs", type=(int,int,int))
def create_sbatch(job_name, job_dir, cluster, partition, time, test, prog, branch_n, branch_i, paths, options, sbatch_dir, loop_epochs):
    job_dir = Path(job_dir) / job_name
    job_dir.mkdir(exist_ok=True, parents=True)

    prog_dir = Path(__file__).parent.absolute()
    inceptiontime_dir = 'TEST'
    
    if not test:
        # clone n_track_ML
        repo = "git@github.com-n_track_ML:hajaalin/n_track_ML.git"
        prog_dir = str(git_clone(repo, branch_n) / "scripts" / "tsc")

        # clone InceptionTime
        repo = "git@github.com-InceptionTime:hajaalin/InceptionTime.git"
        inceptiontime_dir = str(git_clone(repo, branch_i))        
        
    values = {'job_name': job_name, \
              'job_dir': str(job_dir), \
              'cluster': cluster, \
              'partition': partition, \
              'time': time, \
              'prog_dir': prog_dir, \
              'prog': prog, \
              'inceptiontime_dir': inceptiontime_dir, \
              'paths': paths, \
              'options': options, \
    }

    sbatch_dir = Path(sbatch_dir)
    sbatch_dir.mkdir(exist_ok=True, parents=True)
    
    # add a common timestamp to all subtasks
    now = datetime.now().strftime("%Y%m%d%H%M")

    if loop_epochs:
        emin,emax,edelta = loop_epochs
        assert not "--epochs" in options, "--epochs conflicts with --loop_epochs."


        # remember original options
        for epochs in range(emin,emax,edelta):
            print("epochs: " + str(epochs))
            values['options'] = options + " --epochs=" + str(epochs) + " --now=" + now + " --inceptiontime_dir=" + inceptiontime_dir
            
            sbatch = Environment().from_string(BASH).render(values)
            filename = "sbatch_" + job_name + "_e" + str(epochs) + ".sh"

            with open(sbatch_dir / filename, 'w') as f:
                print(sbatch, file=f)

    else:
        values['options'] = options + " --now=" + now + " --inceptiontime_dir=" + inceptiontime_dir
        sbatch = Environment().from_string(BASH).render(values)
        filename = "sbatch_" + job_name + ".sh"

        with open(sbatch_dir / filename, 'w') as f:
            print(sbatch, file=f)

    print('Done.')

if __name__ == "__main__":
    create_sbatch()
    
