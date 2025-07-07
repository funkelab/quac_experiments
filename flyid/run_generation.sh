# We will run generation from everything to everything else.
for source_class in 11  15 17;
do
    for target_class in 11 15 17
    do
        if [ "$source_class" -eq "$target_class" ]; then
            continue
        fi
        echo "Generating images from $source_class to $target_class"
        bsub -n 12 -gpu "num=1" -q gpu_a100 -o "worker_logs/generation_from_${source_class}_to_${target_class}.log" pixi run python 03_generate_images.py --source_class $source_class --target_class $target_class -c config.yaml --checkpoint_iter 40000
    done
done