STOP = ["Observation:", "<Solution complete>"]
# CODE_LTAG = "```python"
# CODE_RTAG = "```\n"
OBSERVATION_LTAG = "Observation: "
OBSERVATION_RTAG = ""

TOOL_DESC_TOOLQA = """(1) RetrieveAgenda[keyword], which retrieves the agenda related to keyword.
(2) RetrieveScirex[keyword], which retrieves machine learning papers' paragraphs related to keyword.
(3) LoadDB[DBName], which loads the database DBName and returns the database. The DBName can be one of the following: flights/coffee/airbnb/yelp.
(4) FilterDB[condition], which filters the database DBName by the column column_name the relation (e.g., =, >, etc.) and the value value, and returns the filtered database.
(5) GetValue[column_name], which returns the value of the column column_name in the database DBName.
(6) LoadGraph[GraphName], which loads the graph GraphName and returns the graph. The GraphName can be one of the following: PaperNet/AuthorNet.
(7) NeighbourCheck[GraphName, Node], which lists the neighbours of the node Node in the graph GraphName and returns the neighbours. 
(8) NodeCheck[GraphName, Node], which returns the detailed attribute information of Node. 
(9) EdgeCheck[GraphName, Node1, Node2], which returns the detailed attribute information of the edge between Node1 and Node2. 
(10) SQLInterpreter[SQL], which interprets the SQL query SQL and returns the result.
(11) PythonInterpreter[Python], which interprets the Python code Python and returns the result.
(12) Finish[answer], which returns the answer and finishes the task."""

# 2. If some steps require the use of tools, you should accurately specify the tool names as well as the settings of the parameters.
# 3. You may encounter tool invocation failures. Please pay attention to the error messages to correct the use of the tool.

custom_prefix = """You are a powerful agent with excellent capabilities in using tools. Answer the questions as best you can. You have access to the following tool:

{tool_desc}

Please adhere to the guidelines as follows:
1. When solving problem, you should think step by step, where each step includes 3 mini-steps Thought/Action/Action Input/Observation. 
2. If some steps require the use of tools, you should accurately specify the tool names as well as the settings of the parameters.
3. If some step requires accurate calculation, you should write Python code and execute for accurate result.
4. Upon completing the task, you should call Finish[answer] to return the answer.
5. Please use the following template.

Question: the input question

Thought: the text analysis

Action: the action to take, should be tool_names

Action Input: the parameters of tools, should be in JSON format

Observation: the result of the tool invocation.

... (this Thought/Action/Action Input/Observation can repeat N times)

"""


custom_suffix = """Now! It's your turn.

Question: {input}"""
