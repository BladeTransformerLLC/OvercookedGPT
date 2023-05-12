
def agent_settings(arglist, agent_name):
    if agent_name[-1] == "1": return arglist.model1
    elif agent_name[-1] == "2": return arglist.model2
    elif agent_name[-1] == "3": return arglist.model3
    elif agent_name[-1] == "4": return arglist.model4
    else: raise ValueError("Agent name doesn't follow the right naming, `agent-<int>`")

class colors:
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"


import re
__code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)
def extract_python_code(content: str) -> str:
    global __code_block_regex
    code_blocks: list = __code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n"
        for block in code_blocks:
            if block.startswith("python"):
                full_code += block[7:] + "\n"
            elif block.startswith(" python"):
                full_code += block[8:] + "\n"
            else:
                #pass
                full_code += block[0:] + "\n"
        print(colors.GREEN + "\n=========== execution =============")
        print(full_code)
        print("===================================" + colors.ENDC)
        return full_code
    else:
        return None
