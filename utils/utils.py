
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

