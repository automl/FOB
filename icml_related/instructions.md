# How to run imagenet experiment with FOB
Assumes you have a conda environment with python >= 3.10
```bash
git clone git@github.com:automl-private/FOB.git
cd FOB
git checkout dev
pip install -r requirements.txt
pip install pytorch_cpr
mkdir submissions/adamcpr
cp <path/to/submission.py> submissions/adamcpr/submission.py

```
cache dir: /p/fastdata/mmlaion/franke5/model/cache/fob

cd /p/scratch/laionize/franke5/workspace/ICML2024_experiments

ml Stages/2024
ml CUDA/12
ml GCC/12.3.0
ml Python/3.11.3
source /p/scratch/laionize/franke5/pt21/bin/activate

ml Stages/2024
ml CUDA/12
ml GCC/12.3.0
ml Python/3.11.3
source /p/scratch/laionize/franke5/pt21r/bin/activate


Dataset setup:
```
python dataset_setup.py -w imagenet64 -d /p/fastdata/mmlaion/franke5/model/cache/fob
```
Where `<DATA_DIR>` will be the folder where the data is stored.

The command to run the workload is as follows:
```
srun python submission_runner.py --workload=imagenet64 --output=<OUTPUT_DIR> --data_dir=<DATA_DIR> --submission=<SUBMISSION> --hyperparameters <HYPERPARAMETERS> --workers=<WORKERS> --seed <SEED> --start_trial $SLURM_ARRAY_TASK_ID --start_hyperparameter $SLURM_ARRAY_TASK_ID --silent
```
Where
- `<OUTPUT_DIR>` is the folder where results will be stored
- `<DATA_DIR>` is the same as before
- `<SUBMISSION>` is either `adamw_baseline` or `adamcpr`
- `<HYPERPARAMETERS>` is the path to the folder with the hyperparameters matching the `<SUBMISSION>`
- `<WORKERS>` matches the cpus available per gpu
- `<SEED>` is different from 42

Make sure to match the SLURM array range to the number of hyperparameters.
