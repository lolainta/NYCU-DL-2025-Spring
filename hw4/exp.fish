#/usr/bin/fish

function submit -a name msg
    set -l name $argv[1]
    set -l msg $argv[2]
    set -l path (realpath ./log/$name/val)
    set -l submission_path "$path/submission.csv"

    echo "Submitting to Kaggle..."
    kaggle competitions submit -c nycu-dlp-2025-spring-lab-4 -f $submission_path -m "$msg" --quiet
    echo "Submission path: $submission_path"
    echo "Message: $msg"
end

function trainer
    set -l name $argv[1]
    set -l path (realpath $name)
    echo "Starting training for $name at $(date +%Y-%m-%d_%H-%M-%S)"
    echo "Training command: python3 ./Trainer.py --save_root log/$name/train --DR dataset/ $argv[2..-1]"
    python3 ./Trainer.py --save_root log/$name/train --DR dataset/ $argv[2..-1]
    echo "Training complete for $name at $(date +%Y-%m-%d_%H-%M-%S)"
end

function tester
    set -l name $argv[1]
    set -l path (realpath $name)
    echo "Starting testing for $name at $(date +%Y-%m-%d_%H-%M-%S)"
    echo "Testing command: python3 ./Tester.py --save_root log/$name/val --DR dataset/ --ckpt_path log/$name/train/best.ckpt"
    python3 ./Tester.py --save_root log/$name/val --DR dataset/ --ckpt_path log/$name/train/best.ckpt
    echo "Testing complete for $name at $(date +%Y-%m-%d_%H-%M-%S)"
end

function exp1
    trainer tfr1 --tfr 1.0 --tfr_sde 0 --tfr_d_step 0.1
    tester tfr1
    submit tfr1 "tfr1: 1.0, 0, 0.1"

    trainer tfr2 --tfr 0.5 --tfr_sde 10 --tfr_d_step 0.05
    tester tfr2
    submit tfr2 "tfr2: 0.5, 10, 0.05"

    trainer tfr3 --tfr 0 --tfr_sde 0 --tfr_d_step 0.1
    tester tfr3
    submit tfr3 "tfr3: 0, 0, 0.1"
end

function exp2
    trainer kl1 --kl_anneal_type cyclical
    tester kl1
    submit kl1 "kl1: cyclical"

    trainer kl2 --kl_anneal_type linear
    tester kl2
    submit kl2 "kl2: linear"

    trainer kl3 --kl_anneal_type constant
    tester kl3
    submit kl3 "kl3: constant"

    trainer kl4 --kl_anneal_type cyclical --kl_anneal_ratio 0.5
    tester kl4
    submit kl4 "kl4: cyclical, 0.5"

end
