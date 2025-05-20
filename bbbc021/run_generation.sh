# bsub -I -n 12 -gpu "num=1" -q gpu_a100 pixi run python 02_generate_images.py --source_class dmso --target_class microtubule_destabilizers --checkpoint_iter 29000 # -o "worker_logs/generation_from_dmso_to_${class_name}.log" -e "worker_logs/generation_from_dmso_to_${class_name}.err"

# We will run generation from DMSO to everything, and from everything to DMSO 
for class_name in actin_disruptors microtubule_destabilizers protein_degradation microtubule_stabilizers protein_synthesis; 
do
    echo "Generating images from DMSO to $class_name"
    bsub -n 12 -gpu "num=1" -q gpu_a100 -o "worker_logs/generation_from_dmso_to_${class_name}.log" -e "worker_logs/generation_from_dmso_to_${class_name}.err" pixi run python 02_generate_images.py --source_class dmso --target_class $class_name --checkpoint_iter 29000 
    echo "Generating images from $class_name to DMSO"
    bsub -n 12 -gpu "num=1" -q gpu_a100 -o "worker_logs/generation_from_${class_name}_to_dmso.log" -e "worker_logs/generation_from_${class_name}_to_dmso.err" pixi run python 02_generate_images.py --source_class $class_name --target_class dmso --checkpoint_iter 29000 
done
