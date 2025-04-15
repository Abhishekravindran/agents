from typing import List
import getpass
import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
from copy import deepcopy
from functools import partial
import dotenv
import logging

from agent.expel import ExpeL
from agent.react import ReAct
from prompts.templates.system import system_message_prompt
from prompts.templates.human import HUMAN_CRITIQUES
from prompts import (
    SYSTEM_INSTRUCTION,
    HUMAN_INSTRUCTION,
    FEWSHOTS,
    REFLECTION_FEWSHOTS,
    HUMAN_REFLECTION_INSTRUCTION,
    SYSTEM_REFLECTION_INSTRUCTION,
    SYSTEM_CRITIQUE_INSTRUCTION,
    RULE_TEMPLATE,
    LLM_PARSER,
    OBSERVATION_FORMATTER,
    STEP_IDENTIFIER,
    CYCLER,
    STEP_CYCLER,
    REFLECTION_PREFIX,
    PREVIOUS_TRIALS_FORMATTER,
    STEP_STRIPPER,
    CRITIQUE_SUMMARY_SUFFIX,
)
from envs import ENVS, INIT_TASKS_FN
from memory import (
    EMBEDDERS,
    RETRIEVERS,
)
from models import LLM_CLS
from utils import get_fewshot_max_tokens, load_trajectories_log, save_trajectories_log, split_logs_by_task, plot_trial_stats, alfworld_results_per_env_name_log, get_webshop_mean_score, get_split_eval_idx_list, set_seed, get_env


def get_eval_num(eval_idx: int, eval_idx_list: List[List[int]]) -> int:
    eval_num = 0
    for eval_idxs in eval_idx_list:
        if eval_idx in eval_idxs:
            break
        eval_num += len(eval_idxs)
    return eval_num + eval_idxs.index(eval_idx)


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Use Ollama by default
    cfg.llm = "ollama"
    
    # Create environment
    env = get_env(cfg.env_name)
    
    # Initialize agent
    if cfg.agent == 'expel':
        agent = ExpeL(
            env=env,
            llm=cfg.llm,
            openai_api_key=None,  # Not needed for Ollama
            max_steps=cfg.max_steps,
            max_iterations=cfg.max_iterations,
            max_retries=cfg.max_retries,
            verbose=cfg.verbose,
        )
    elif cfg.agent == 'react':
        agent = ReAct(
            env=env,
            llm=cfg.llm,
            openai_api_key=None,  # Not needed for Ollama
            max_steps=cfg.max_steps,
            max_retries=cfg.max_retries,
            verbose=cfg.verbose,
        )
    else:
        raise ValueError(f'Unknown agent: {cfg.agent}')
    
    LOG_PATH = Path('/'.join([cfg.log_dir, cfg.benchmark.name, cfg.agent_type]))
    SAVE_PATH = LOG_PATH / 'eval'
    SAVE_PATH.mkdir(exist_ok=True)

    print(f"{SAVE_PATH}/{cfg.run_name}.pkl")
    
    # Overwriting confirmation
    if not cfg.resume and os.path.exists(f"{SAVE_PATH}/{cfg.run_name}.pkl") and cfg.run_name != 'test':
        while True:
            res = input(f"Are you sure to overwrite '{cfg.run_name}'? (Y/N)\n").lower()
            if res == 'n':
                exit(0)
            elif res == 'y':
                break

    # Load trajectory checkpoint
    out = load_trajectories_log(
        SAVE_PATH if cfg.resume else LOG_PATH,
        run_name=cfg.load_run_name,
        load_log=cfg.resume,
        load_true_log=cfg.resume)
    dicts = out['dicts']
    eval_idx_list = dicts[-1].get(
        'eval_idx_list',
        get_split_eval_idx_list(dicts[-1], cfg.benchmark.eval_configs.k_folds))
    log = out['log'] if cfg.resume else f'### EVAL ORDER ###\n{eval_idx_list}\n'
    true_log = out['true_log'] if cfg.resume else f'### EVAL ORDER ###\n{eval_idx_list}\n{str(cfg)}\n'

    num_training_tasks = len(INIT_TASKS_FN[cfg.benchmark.name](cfg))

    # we start at fold 0 if we are starting a new run
    starting_fold = dicts[-1].get('starting_fold', 0)
    # we start at the first task in the fold if we are starting a new run
    starting_idx = dicts[-1].get('starting_idx', eval_idx_list[0][0])

    react_agent = agent

    if len(dicts) > 0:
        no_load_list = ['ai_message', 'message_type_format', 'max_num_rules', 'testing', 'human_critiques', 'system_critique_instructions', 'fewshot_strategy', 'success', 'halted', 'fail', 'task_idx', 'prompt_history', 'critique_truncate_strategy', 'success_critique_num', 'reflection_fewshots', 'reflection_system_prompt', 'reflection_prefix', 'reflection_prompt_history', 'reflections', 'previous_trial', 'perform_reflection', 'increment_task', 'reflection_system_kwargs', 'prepend_human_instruction', 'name', 'tasks', 'human_instruction_kwargs', 'all_system_instruction', 'all_fewshots', 'max_steps', 'ordered_summary', 'fewshots', 'system_instruction', 'num_fewshots', 'curr_step', 'log_idx', 'pretask_idx', 'reflect_interaction_idx', 'truncated', 'reward', 'terminated', 'autoregressive_model_instruction', 'failed_training_task_idx', '_train', 'task',
        'eval_idx_list', 'starting_fold', 'starting_idx', 'rule_template', 'max_fewshot_tokens', 'buffer_retrieve_ratio']
        react_agent.load_checkpoint(dicts[-1], no_load_list=no_load_list)
        # resetting task_idx
        react_agent.task = react_agent.tasks[starting_idx]['task']
        react_agent.reset()

    react_agent.eval()
    start_processing = False # Flag for starting_fold
    start_eval_idx = False # Flag for starting_idx
    first_flag = True
    react_agent.no_rules = cfg.no_rules

    print(f'*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n\nWe are using the following model: {react_agent.llm.model_name}\n\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    true_log += str(react_agent.llm.llm) + '\n'
    first_idxs = [eval_idxs[0] for eval_idxs in eval_idx_list]

    # start evaluating
    for k, eval_idxs in enumerate(eval_idx_list):
        # syncing fold
        if k == starting_fold or start_processing:
            start_processing = True
            if not cfg.no_rules:
                # make sure every start of fold does create_rules
                if not first_flag:
                    starting_idx = eval_idxs[0]
                # create rules for each fold, if starting a new fold
                if starting_idx == eval_idxs[0]:
                    training_ids = set(range(num_training_tasks)) - set(eval_idxs)
                    react_agent.create_rules(
                        list(training_ids),
                        cache_fold=None,
                        load_cache_fold=k if cfg.load_cache_rules else None,
                    )
                first_flag = False

            # evaluate on each task in the fold
            for eval_idx in eval_idxs:
                # syncing idx
                if eval_idx == starting_idx or start_eval_idx:
                    start_eval_idx = True
                    # Skip the first matching eval_idx
                    if eval_idx == starting_idx and starting_idx not in first_idxs:
                        continue
                    prefix = f"#######################################\nTASK {get_eval_num(eval_idx, eval_idx_list)} \nFOLD: {k}, EVAL_IDX: {eval_idx}\n" # the space after TASK \d+ is needed for log results parsing
                    prefix += react_agent.remove_task_suffix(react_agent.tasks[eval_idx]['task']) + '\n'
                    print(prefix)

                    react_agent.run(mode='eval', eval_idx=eval_idx)

                    # logging
                    react_agent.update_stats()
                    log += prefix + react_agent.log_history(include_task=False) + '\n\n'
                    true_log += prefix + react_agent.log_history(include_all=True, include_task=False) + '\n\n'
                    # not saving other complicated objects
                    eval_dict = {k: deepcopy(v) for k, v in react_agent.__dict__.items() if type(v) in [list, set, str, bool, int, dict]}
                    eval_dict.update({
                        'eval_idx_list': eval_idx_list,
                        'starting_fold': k,
                        'starting_idx': eval_idx,
                    })
                    dicts.append(eval_dict)
                    save_trajectories_log(
                        path=SAVE_PATH, 
                        log=log, 
                        dicts=dicts,
                        true_log=true_log,
                        run_name=f'{cfg.run_name}'
                    )

    # logging to files
    success, fail, halted = react_agent.get_stats()
    log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"
    true_log += f"########################################\nEND TRIAL\nTrial summary: Success: {success}/{success + fail + halted}, Fail: {fail}/{success + fail + halted}, Halted: {halted}/{success + fail + halted}"

    print(f'Finished. Success: {success}, Fail: {fail}, Halted: {halted}')

    parsed_result = split_logs_by_task(text=log, num_tasks=len(react_agent.tasks))
    reflection_results = plot_trial_stats(parsed_result=parsed_result, benchmark=cfg.benchmark.name, max_trials=1, save_path=f"{LOG_PATH}/{cfg.run_name}_logs_stats.png")

    results = ', '.join([f"{k}: {v}" for k, v in reflection_results.items()]) + '\n'
    if cfg.benchmark.name == 'alfworld':
        results += str(alfworld_results_per_env_name_log(log, len(react_agent.tasks), 1))
    elif cfg.benchmark.name == 'webshop':
        results += str(get_webshop_mean_score(log, len(react_agent.tasks), 1))
    log += f'\n\n{results}\n########################################'
    true_log += f'\n\n{results}\n########################################'
    print(results)

    save_trajectories_log(
        path=SAVE_PATH, 
        log=log, 
        dicts=dicts,
        true_log=true_log,
        run_name=f'{cfg.run_name}'
    )

if __name__ == "__main__":
    main()
