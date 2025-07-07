for name in DIntegratedGradients DDeepLift VanillaIntegratedGradients VanillaDeepLift; do
    echo "Running evaluation for $name"
    bsub -n 12 -gpu "num=1" -q gpu_a100 -o worker_logs/evaluation_$name.log pixi run python 05_evaluate.py -n $name -c config.yaml
done 