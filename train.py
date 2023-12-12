import torch.multiprocessing as mp
from agent import Agent
from process import process

def train(arguments):
    learning_rate = arguments.lr
    gamma = arguments.gamma
    n_updates = arguments.epochs
    clip = arguments.clip
    c1 = arguments.c1
    c2 = arguments.c2
    minibatch_size = arguments.minibatch_size
    batch_size = arguments.batch_size
    episodes = arguments.episodes
    lam = arguments.lam
    in_channels = arguments.in_channels
    n_outputs = arguments.n_outputs
    num_processes = arguments.num_processes
    alpha = arguments.alpha
    beta = arguments.beta
    env_name = arguments.env
    is_global = True

    mp.set_start_method('spawn')

    global_agent = Agent(in_channels, 
        n_outputs, 
        learning_rate, 
        n_updates, 
        clip, 
        minibatch_size, 
        c1, 
        c2, 
        alpha, 
        beta, 
        is_global
    )

    processes = []
    for thread in range(num_processes):
        p = mp.Process(
            target=process, 
            args=(
                thread,
                n_outputs,
                episodes,
                batch_size,
                lam,
                in_channels,
                learning_rate,
                gamma,
                n_updates, 
                clip,
                minibatch_size,
                c1,
                c2,
                env_name,
                alpha, 
                beta,
                global_agent
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    

