import numpy as np
import torch
import torch.nn.functional as F
import ast
import concurrent.futures

from utils.chat_utils import chat_with_gpt
from agents.system_prompt import (
    Instruction_system_prompt_Generator_correct,
    Instruction_system_prompt_Generator_incorrect,
    Instruction_system_prompt_Discriminator,
    Parsing_prompt,
    System_prompt_Generator_correct,
    System_prompt_Generator_incorrect,
    System_prompt_Discriminator,
    )


class Equilibrium_Ranking():
    r""" EQUILIBRIUM - RANKING : LM R ANKING VIA EQUILIBRIUM SEARCH
    1. generate initial policies (only focus on the signalling policies)
    2. update policies using No-regret learning algorithms
    3. make decision based on the learned policies
    
    Args:
    """
    def __init__(self, text_queries, object_category):
        
        # Hyperparameters
        self.lambda_G = 0.1
        self.lambda_D = 0.1
        self.eta_G = 0.1
        self.eta_D = 0.1
        self.iterations = 500
        
        target_hint = "Hint: the target category is " + object_category
        
        # self.chat_history_for_generator_correct = []
        # self.chat_history_for_generator_incorrect = []
        # self.chat_history_for_generator_discriminator = []
        
        # self.chat_history_for_generator_correct.append({"role": "system", "content": Instruction_system_prompt_Generator_correct})
        
        # self.chat_history_for_generator_correct.append({"role": "user", "content": text_queries})
        
        # self.response_message = chat_with_gpt(self.chat_history_for_generator_correct, 2)
        # self.chat_history_for_generator_correct.append({"role": "assistant", "content": self.response_message})
        
        # ground_history = self.chat_history_for_generator_correct[-2:]
        
        # self.chat_history_for_generator_incorrect.append({"role": "system", "content": Instruction_system_prompt_Generator_incorrect})
        # self.chat_history_for_generator_incorrect.extend(ground_history)
        # self.chat_history_for_generator_discriminator.append({"role": "system", "content": Instruction_system_prompt_Discriminator})
        # self.chat_history_for_generator_discriminator.extend(ground_history)
        
        self.text_description = text_queries + target_hint
        message = []
        message.append({"role": "system", "content": Parsing_prompt})
        message.append({"role": "user", "content": self.text_description})
        self.response_message = chat_with_gpt(message, 2)
        
    def equilibrium_search(self, user_prompt, cadidate_num):
        self.candidate_num = cadidate_num
        chat_history_for_generator_correct = []
        chat_history_for_generator_discriminator = []
        
        chat_history_for_generator_correct.append({"role": "system", "content": System_prompt_Generator_correct})
        chat_history_for_generator_discriminator.append({"role": "system", "content": System_prompt_Discriminator})
        
        user_prompt = self.text_description + user_prompt
        # generate initial policies 
        # Get generative probabilities for "correct"
        chat_history_for_generator_correct.append({"role": "user", "content": user_prompt})
        generative_probabilities_correct = self._get_generative_probabilities(chat_history_for_generator_correct)
        
        # Get generative probabilities for "incorrect"
        # self.chat_history_for_generator_incorrect.append({"role": "user", "content": user_prompt})
        # generative_probabilities_incorrect = self._get_generative_probabilities(self.chat_history_for_generator_incorrect)
        generative_probabilities_incorrect = [1 - value for value in generative_probabilities_correct]
        
        # Get discriminative probabilities for each candidate
        chat_history_for_generator_discriminator.append({"role": "user", "content": user_prompt})
        discriminative_probabilities = self._get_discriminative_probabilities(chat_history_for_generator_discriminator)
        
        plm_y_given_x_v = {
            'correct': generative_probabilities_correct,
            'incorrect': generative_probabilities_incorrect
        }
        plm_v_given_x_y = discriminative_probabilities
        
        
        pi_G_1 = self._normalize_initial_policy(plm_y_given_x_v, axis=0)
        pi_D_1 = self._normalize_initial_policy(plm_v_given_x_y, axis=0)

        # Print initial policies
        print("Initial Generator Policy:", pi_G_1)
        print("Initial Discriminator Policy:", pi_D_1)
        
        v_values = ['correct', 'incorrect']
        y_values = list(plm_v_given_x_y.keys())
        
        # Initialize Q values
        Q_G = {v: np.zeros(len(y_values)) for v in v_values}
        Q_D = {y: np.zeros(len(v_values)) for y in y_values}
        
        # Equilibrium ranking
        for t in range(1, self.iterations + 1):
            # Update Q values
            for v in v_values:
                for i, y in enumerate(y_values):
                    Q_G[v][i] += (1 / (2 * t)) * pi_D_1[y][v_values.index(v)]

            for y in y_values:
                for i, v in enumerate(v_values):
                    Q_D[y][i] += (1 / (2 * t)) * pi_G_1[v][y_values.index(y)]

            # Update policies
            pi_G = self._update_policy(Q_G, pi_G_1, self.lambda_G, self.eta_G, t)
            pi_D = self._update_policy(Q_D, pi_D_1, self.lambda_D, self.eta_D, t)

        # Display the final policies
        print("Final Generator Policy:", pi_G)
        print("Final Discriminator Policy:", pi_D)
        
        return pi_G['correct'].argmax(axis=0)
                        
    def generator_search(self, user_prompt):
        chat_history_for_generator_correct = []
        
        chat_history_for_generator_correct.append({"role": "system", "content": System_prompt_Generator_correct})
        
        user_prompt = self.text_description + user_prompt
        # generate initial policies 
        chat_history_for_generator_correct.append({"role": "user", "content": user_prompt})
        
        generative_probabilities_correct = self._get_generative_probabilities(chat_history_for_generator_correct, num_samples=1)
        
        return generative_probabilities_correct.argmax(axis=0)

    # Normalizing initial policies
    def _normalize_initial_policy(self, plm_dict, axis=0):
        norm_dict = {}
        for key in plm_dict:
            norm_dict[key] = plm_dict[key] / np.sum(plm_dict[key], axis=axis)
        return norm_dict
    
    # Function to update policies
    def _update_policy(self, Q, pi, lambd, eta, t):
        updated_pi = {}
        for key in Q:
            exponent = (Q[key] + lambd * np.log(pi[key])) / (1 / (eta * t) + lambd)
            updated_pi[key] = np.exp(exponent) / np.sum(np.exp(exponent))
        return updated_pi

    def _get_generative_probabilities(self, chat_history, num_samples=5):
        probabilities = []
        responses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_message = {executor.submit(chat_with_gpt, chat_history): chat_history for _ in range(num_samples)}
                for future in concurrent.futures.as_completed(future_to_message):
                    prompt = future_to_message[future]
                    try:
                        response = future.result()
                        responses.append(response)
                    except Exception as exc:
                        print(f'{prompt} generated an exception: {exc}')
                
        for i, response in enumerate(responses):
            ground_json = ast.literal_eval(response)
            probabilities_list = np.array(list(ground_json["objects_scores"].values()) )
            if len(list(ground_json["objects_scores"].values()) ) == self.candidate_num + 1:
                probabilities.append(probabilities_list/np.sum(probabilities_list))
        return np.sum(np.stack(probabilities), axis = 0)
        
    def _get_discriminative_probabilities(self, chat_history, num_samples=5):
        probabilities = {}
        responses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_prompt = {executor.submit(chat_with_gpt, chat_history): chat_history for _ in range(num_samples)}
                for future in concurrent.futures.as_completed(future_to_prompt):
                    prompt = future_to_prompt[future]
                    try:
                        response = future.result()
                        responses.append(response)
                    except Exception as exc:
                        print(f'{prompt} generated an exception: {exc}')
                        
        for i, response in enumerate(responses):
            ground_json = ast.literal_eval(response)
            discriminate = ground_json["objects_discriminate"] 
            for key, value in discriminate.items():
                if value == "accept":
                    if key not in probabilities:
                        probabilities[key] = 1 
                    else:
                        probabilities[key] += 1
                else:
                    if key not in probabilities:
                        probabilities[key] = 0
                    else:
                        probabilities[key] += 0
        return {key:[value/num_samples, 1-value/num_samples] for key, value in probabilities.items()}
    
    



