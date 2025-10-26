import json
import re

def extract_think_format(llm_solution: str):
    think_pair_count = len(re.findall(r'<think>(.*?)</think>', llm_solution, re.DOTALL))
    think_left_count = len(re.findall(r'<think>', llm_solution, re.DOTALL))
    think_right_count = len(re.findall(r'</think>', llm_solution, re.DOTALL))

    bad_pattern = re.search(r'<think>\n ', llm_solution) # verl sglang multi_turn bug: extra space after think.

    if think_pair_count == think_left_count and think_left_count == think_right_count and not bad_pattern:
        return True
    else:
        return False

def extract_tool_format(llm_solution: str):
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_calls = re.findall(tool_call_pattern, llm_solution, re.DOTALL)
    
    tool_response_pattern = r'<tool_response>(.*?)</tool_response>'
    tool_responses = re.findall(tool_response_pattern, llm_solution, re.DOTALL)

    error_count = 0
    for tool_response in tool_responses:
        if 'Tool call execute failed, exception message:' in tool_response: # format error
            return False
        elif re.search(r"No .* named .* found\.", tool_response): # search error
            error_count += 1
        elif re.search(r"No edit was performed\.", tool_response): # edit error
            error_count += 1
    
    tool_call_count = len(tool_calls)
    tool_response_count = len(tool_responses)

    unique_tool_call_count = len(set(tool_calls))
    unique_tool_response_count = len(set(tool_responses))

    tool_call_left_count = len(re.findall(r'<tool_call>', llm_solution, re.DOTALL))
    tool_call_right_count = len(re.findall(r'</tool_call>', llm_solution, re.DOTALL))

    tool_response_left_count = len(re.findall(r'<tool_response>', llm_solution, re.DOTALL))
    tool_response_right_count = len(re.findall(r'</tool_response>', llm_solution, re.DOTALL)) 
    if tool_call_count == 0 and tool_response_count == 0: # do not use tool
        return False
    elif tool_call_count != tool_response_count: # call and response do not match
        return False
    elif tool_call_count != unique_tool_call_count or tool_response_count != unique_tool_response_count: # repetitive calls
        return False
    elif tool_call_count == error_count: # all calls failed
        return False
    elif tool_call_left_count != tool_call_right_count or tool_response_left_count != tool_response_right_count: # tooluse unpair
        return False
    else:
        try:
            for i in range(len(tool_calls)):
                call_json = tool_calls[i].strip()
                call_dict = json.loads(call_json)
                if call_dict['name'] == 'search_tool': # search only (mandatory)
                    construct = call_dict['arguments']['construct']
                    if construct in ["function", "class", "class_method"] and not re.search(r"No .* named .* found\.", tool_responses[i].strip()):
                        return True
            return False
        except Exception as e:
            print(f"Error occurred when extracting tool format: {str(e)}")
            return False

def extract_patch(llm_solution: str):
    patch_pattern = r'\[PATCH\]\n(.*?)\n\[/PATCH\]'
    patch_match = re.findall(patch_pattern, llm_solution, re.DOTALL)
    if patch_match:
        return patch_match[-1].strip() # use final patch
    else:
        return "" # no patch
        
def extract_answer_naive(llm_solution: str):
    answer_pattern = r'\\boxed\{(.*?)\}'
    answer_match = re.findall(answer_pattern, llm_solution)
    if answer_match:
        return answer_match[-1].strip() # use final answer
    else:
        return None # no answer

def extract_answer_vm(llm_solution: str):
    answer_pattern = r'<judgement>(.*?)</judgement>'
    answer_match = re.findall(answer_pattern, llm_solution)
    if answer_match:
        answer_str = answer_match[0].strip() # use first answer
        if answer_str == 'YES':
            return 'true'
        elif answer_str == 'NO':
            return 'false'
        else:
            return None # invalid answer
    else:
        return None # no answer

def extract_answer_rm(llm_solution: str):
    answer_pattern = r'\[\[(.*?)\]\]'
    answer_match = re.findall(answer_pattern, llm_solution)
    if answer_match:
        return answer_match[-1].strip() # use final answer
    else:
        return None # no answer

def extract_answer_pair(llm_solution: str):
    answer_pattern = r'\\boxed\{(.*?)\}'
    answer_match = re.findall(answer_pattern, llm_solution)
    if answer_match:
        return answer_match[-1].strip() # use final answer
    else:
        return None # no answer

def extract_batch_combine(llm_solution: str):
    answer = None
    # answer_pattern = r'\\boxed\{(\d*?)\}'
    answer_pattern = r'\\boxed\{(.*?)\}'
    answer_match = re.findall(answer_pattern, llm_solution)
    # if answer_match and len(answer_match) == 1:
    if answer_match:
        answer = answer_match[-1].strip() # use final answer
    else:
        return None # no answer

    if ', ' in answer: # fuzz match
        answer = answer.split(', ')
    elif ',' in answer:
        answer = answer.split(',')
    else:
        answer = list(answer)

    if len(answer) > 4:
        return None # invalid length
    else:
        try:
            answer_list = [int(i) for i in answer]
        except:
            return None # invalid ID format
        for item in answer_list:
            if item < 1 or item > 4:
                return None # invalid candidate number
        answer_set = set(answer_list)
        if len(answer_list) == len(answer_set):
            return [True if i in answer_set else False for i in range(1, 5)]
        else:
            return None # duplicated