def batch_main(config, questions, answer=None, func_exact_match="exact_match"):
   
    # import pdb;pdb.set_trace() 
    answer_dict = {}
    for ques, ans in zip(questions, answer):
        answer_dict[ques] = ans
    
    try:
        with Manager() as manager:
            public_prompts = manager.dict()
            public_outputs = manager.dict()
            public_values = manager.dict()
            public_stopwords = manager.dict()
            public_n = manager.dict()
            public_temperature = manager.dict()
            public_final_outputs = manager.list()
            task_flag = manager.Value('i', 0)
            
            llm_process = Process(target=llm_generate, args=(config, task_flag, public_prompts, public_stopwords, public_n, public_temperature, public_outputs, public_values))
            llm_process.start()
            # process_list.append(llm_process)
            epoch_trees = []
            for question in questions:
                epoch_trees.append((config, question, public_final_outputs, public_prompts, public_stopwords, public_n, public_temperature, public_outputs, public_values, task_flag))
            
            progress_bar_process = Process(target=progress_bar, args=(public_final_outputs, len(epoch_trees)))
            progress_bar_process.start()

            with ProcessPool(max_workers=min(len(epoch_trees), os.cpu_count())) as pool:
                future = pool.map(mcts_search, epoch_trees, timeout=TIMEOUT_SECONDS)
                _ = future.result()
            
            llm_process.terminate()
            progress_bar_process.terminate()

            trees = []
            for final_out in public_final_outputs:
                trees.append(final_out)
    except Exception as e:
        print(colored(f"Exception: {e}", "red"))
        return [""] * len(questions)

    return evaluate(trees, answer_dict, get_postprocess_func(func_exact_match))


def progress_bar(public_outputs, total=None, desc="Execute"):

    progress_bar = tqdm(total=total, desc=desc)
    pre_num = 0
    while True:
        time.sleep(1)
        current_num = len(public_outputs)
        progress_bar.update(current_num - pre_num)
        if current_num >= total:
            break
        else:
            pre_num = current_num
    
    progress_bar.close()