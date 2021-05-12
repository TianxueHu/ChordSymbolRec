srun --gres gpu:1 -c 6 -A overcap -p overcap -J "trial" -x ash,calculon,oppy,johnny5,ava,cortana,asimo --pty bash 
# srun --gres gpu:1 -c 6 -p short -J "trial" -x ash,calculon,oppy,johnny5,ava,cortana --pty bash
# srun --gres gpu:1 -c 6 -p long -J "trial" --pty bash
# --pty python main.py


