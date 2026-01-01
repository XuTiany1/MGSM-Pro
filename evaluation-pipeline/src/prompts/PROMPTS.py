cot_prompt_0 = '''
Answer the following. Just enter the final answer as a number. Do not provide units nor reasoning.
Question:
{question}
Answer:
'''

cot_prompt_1='''
The last number in your response should be the final answer without units.
Question: {question}\n Step-by-Step Answer using English:
'''

cot_prompt_2 = '''
Provide a detailed, step-by-step explanation in English to arrive at the correct solution.  
Ensure that the final number in your response is the final answer without units.  
Question: {question}
'''

cot_prompt_3 = '''
Explain your reasoning step by step in clear English to solve the problem.  
Your response should end with the final numerical answer, without including units.  
Question: {question}
'''

cot_prompt_4 = '''
Break down the solution into logical steps, explaining each one in English.  
The last number you write should be the final computed answer, excluding any units.  
Question: {question}
'''

cot_prompt_5 = '''
Walk through the solution in a structured manner, describing each step in English.  
The final answer should be the last number in your response, without any units.  
Question: {question}
'''
