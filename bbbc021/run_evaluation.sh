for name in discriminative_deeplift discriminative_ig; do
    echo "Running evaluation for $name"
    bsub -n 12 -gpu "num=1" -q gpu_a100 -o worker_logs/evaluation_$name.log pixi run python 04_evaluation.py -n $name
done