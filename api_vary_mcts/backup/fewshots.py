TOOLQA_EASY8 = [
"""Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?

Thought: This is a question related to flights. We need to load the flights database.

Action: LoadDB

Action Input: {"DBName": "flights"}

Observation: We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, Distance, Year, Quarter, Month, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, Flight_Number_Marketing_Airline, Tail_Number, Flight_Number_Operating_Airline, DestAirportID, TaxiIn, CRSArrTime, ArrDelay, ArrDel15

Thought: We need to filter the information related to the query.

Action: FilterDB

Action Input: {"filter_condition": "Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO"}

Observation: We have successfully filtered the data (1 row).

Thought: We then need to know the departure delayed time.

Action: GetValue[DepDelay]

Action Input: {"filter_condition": "Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO"}

Observation: -7.0

Thought: We then need to know the arrival delayed time.

Action: GetValue[ArrDelay]

Observation: -17.0

Thought: To compute the extra minutes, we need to subtract the departure delayed time from the arrival delayed time.

Action: PythonInterpreter[# solution in Python:
print((-17)-(-7))]

Observation: -10

Thought: After calculation, we know that the answer is -10.

Action: Finish[-10]
<Solution complete>""",

"""Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Thought: Maybe I can write a python program to solve this problem.

Action: PythonInterpreter[# solution in Python:
# Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
golf_balls_initial = 58
golf_balls_lost_tuesday = 23
golf_balls_lost_wednesday = 2
golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
result = golf_balls_left
print(result)]

Observation: 33

Thought: After execution of Python Code, we know that the answer is 33.

Action: Finish[33]
<Solution complete>""",

"""Question: What is the corresponding Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task?

Thought: The question is asking some performance information about Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task, we need to retrieve some useful information from the scirex database.

Action: RetrieveScirex[Mean_IoU score of the FRRN method on Cityscapes dataset for Semantic_Segmentation task]

Observation: section : Abstract Semantic image segmentation is an essential component of modern autonomous driving systems , as an accurate understanding of the surrounding scene is crucial to navigation and action planning . ..., The two streams are coupled at the full image resolution using residuals . Without additional processing steps and without pretraining , our approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset .

Thought: As the approach achieves an intersection - over - union score of 71.8 % on the Cityscapes dataset, the answer should be 71.8%.

Action: Finish[71.8%]
<Solution complete>""",

"""Question: When was the paper Learning the Principle of Least Action with Reinforcement Learning. published?

Thought: The question is asking some basic information of a dblp paper. We need to load the DBLP graph.

Action: LoadGraph[dblp]

Observation: DBLP graph is loaded.

Thought: The question is asking the published date of a paper, we need to check the node from the PaperNet in DBLP graph.

Action: NodeCheck[PaperNet, Learning the Principle of Least Action with Reinforcement Learning.]

Observation: {'title': 'Learning the Principle of Least Action with Reinforcement Learning.', 'year': 2021, 'venue': {'raw': 'AAAI Spring Symposium - MLPS'}, 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference', 'page_start': '', 'page_end': ''}

Thought: The published date of the paper is 2021.

Action: Finish[2021]
<Solution complete>""",

"""Question: How many collaborators does Chao Zhang have in the DBLP graph?

Thought: The question is asking the number of collaborators of a person, we need load the DBLP graph.

Action: LoadGraph[dblp]

Observation: DBLP graph is loaded.

Thought: The question is asking the collaborators of a person, we need to check the neighboring nodes from the AuthorNet in DBLP graph.

Action: NeighbourCheck[AuthorNet, Chao Zhang]

Observation: ['YUHUI YUAN', 'Rao Fu', 'Lang Huang', 'Weihong Lin', 'X Chen', 'Jingdong Wang']

Thought: The number of collaborators of Chao Zhang is 6.

Action: Finish[6]
<Solution complete>""",

"""Question: How many papers does Chao Zhang and Weihong Lin have in common in the DBLP graph?

Thought: The question is asking the number of common papers of two persons, we need load the DBLP graph.

Action: LoadGraph[dblp]

Observation: DBLP graph is loaded.

Thought: The question is asking the common papers of two persons, we need to check the edges between them from the PaperNet in DBLP graph.

Action: EdgeCheck[PaperNet, Chao Zhang, Weihong Lin]

Observation: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}

Thought: The number of common papers of Chao Zhang and Weihong Lin is 1.

Action: Finish[1]
<Solution complete>""",

"""Question: Where did Stephen's Opera performance take place?

Thought: The question is asking the location of Stephen's Opera performance from agenda.

Action: RetrieveAgenda[Stephen's Opera performance]

Observation: On January 29, 2022, there will be an opera performance at the Lyric Opera House, featuring Stephen. The show will start at 7:00 PM and end at 9:00 PM. It promises to be a wonderful evening of beautiful music and powerful performances in a stunning venue. Come and experience the magic of opera at its finest!

Thought: The event happened in Lyric Opera.

Action: Finish[Lyric Opera]
<Solution complete>""",
    
"""Question: What was the trading volume of coffee on 2000-01-14?

Thought: Maybe I can write a SQL query to solve this problem.

Action: SQLInterpreter[SELECT Volume FROM coffee.coffee_data WHERE Date = '2000-01-14'];

Observation: Volume: 10115

Thought: The volume of coffee on 2000-01-14 is 10115.

Action: Finish[10115]
<Solution complete>""",]

TOOLQA_HARD3 = [
"""Question: What is the average airtime of the flights from PHX to LGA host by Spirit Air Lines?

Thought: This is a question related to flights. We need to load the flights database.

Action: LoadDB[flights]

Observation: We have successfully loaded the flights database, including the following columns: FlightDate, Airline, Origin, Dest, Cancelled, Diverted, CRSDepTime, DepTime, DepDelayMinutes, DepDelay, ArrTime, ArrDelayMinutes, AirTime, CRSElapsedTime, ActualElapsedTime, Distance, Year, Quarter, Month, DayofMonth, DayOfWeek, Marketing_Airline_Network, Operated_or_Branded_Code_Share_Partners, DOT_ID_Marketing_Airline, IATA_Code_Marketing_Airline, Flight_Number_Marketing_Airline, Operating_Airline, DOT_ID_Operating_Airline, IATA_Code_Operating_Airline, Tail_Number, Flight_Number_Operating_Airline, OriginAirportID, OriginAirportSeqID, OriginCityMarketID, OriginCityName, OriginState, OriginStateFips, OriginStateName, OriginWac, DestAirportID, DestAirportSeqID, DestCityMarketID, DestCityName, DestState, DestStateFips, DestStateName, DestWac, DepDel15, DepartureDelayGroups, DepTimeBlk, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrDelay, ArrDel15, ArrivalDelayGroups, ArrTimeBlk, DistanceGroup, DivAirportLandings

Thought: We need to filter the information related to the query.

Action: FilterDB[Origin=PHX, Dest=ATL, Airline=Spirit Air Lines]

Observation: We have successfully filtered the data (16 rows).

Thought: We then obtain all the AirTime values of filtered records.

Action: GetValue[AirTime]

Observation: [247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, nan, 229.0, nan, 259.0, 253.0]

Thought: We then calculate the average of the values. NaN values are ignored.

Action: Calculate(mean(247.0, 253.0, 230.0, 264.0, 254.0, 275.0, 227.0, 258.0, 245.0, 253.0, 242.0, 229.0, 259.0, 253.0))

Observation: 249.214

Thought: The average airtime of the flights from PHX to LGA host by Spirit Air Lines is 249.2.

Action: Finish[249.2]
<Solution complete>""",

"""Question: Can you recommend a Guamanian business with the highest star rating within a 5-mile radius of 6830 Rising Sun Ave?

Thought: This is a question related to business. We need to load the yelp database.

Action: LoadDB[yelp]

Observation: We have successfully loaded the yelp database, including the following columns: business_id, name, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes, categories, hours

Thought: Maybe we can write SQL code to obtain the latitude and longitude of the address.

Action: SQLInterpreter[SELECT latitude, longitude FROM yelp.yelp_data WHERE address='6830 Rising Sun Ave']  

Observation: latitude: 40.05555, longitude: -75.090723 

Thought: Maybe we can write another python code to calculate the coordinates range of 5-mile radius.

Action: PythonInterpreter[import geopy\nimport geopy.distance\nlatitude = 40.05555\nlongitude = -75.090723\n_, lo_max, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=90)\n_, lo_min, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=270)\nla_max, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=0)\nla_min, _, _ = geopy.distance.distance(kilometers=5).destination(point=(latitude, longitude), bearing=180)\nans = (la_max, la_min, lo_max, lo_min)\nprint(ans)]

Observation: (40.10058038657396, 40.010519262352226, -75.03212327463275, -75.14932272536724)

Thought: We then filter the information related to the query in the yelp database.

Action: FilterDB[latitude>=40.010519262352226, longitude>=-75.14932272536724, latitude<=40.10058038657396, longitude<=-75.03212327463275, categories=Guamanian]

Observation: We have successfully filtered the data (2 rows).

Thought: We then obtain all the stars values of filtered records.

Action: GetValue[stars]

Observation: [4.0, 4.5]

Thought: The highest star rating is 4.5. Select the business with the highest star rating.

Action: FilterDB[stars=4.5]

Observation: We have successfully filtered the data (1 row).

Thought: Get the corresponding name.

Action: GetValue[name]

Observation: Damaso Painting

Thought: The business with the highest star rating is Damaso Painting.

Action: Finish[Damaso Painting]
<Solution complete>""",

"""Question: What venue does Chao Zhang with his collarborator publish paper on?

Thought 1: This is a question related to DBLP. We need to load the DBLP graph.
Action: LoadGraph[dblp]

Observation: DBLP graph is loaded.

Thought: We need to check the neighbouring nodes of Chao Zhang in AuthorNet.

Action: NeighbourCheck[AuthorNet, Chao Zhang]

Observation: Weihong Lin

Thought: We need to check the between Chao Zhang and Weihong Lin to check their collaboration.

Action: EdgeCheck[AuthorNet, Chao Zhang, Weihong Lin]

Observation: {'weight': 1, 'papers': ['HRFormer: High-Resolution Vision Transformer for Dense Predict.'], 'n_citation': [95]}

Thought: We need to check the paper detail in the node feature from PaperNet they published.

Action: NodeCheck[PaperNet, HRFormer: High-Resolution Vision Transformer for Dense Predict.]

Observation: {'title': 'HRFormer: High-Resolution Vision Transformer for Dense Predict.', 'year': 2021, 'venue': {'raw': 'AAAI Spring Symposium - MLPS'}, 'n_citation': 0, 'keywords': [], 'doc_type': 'Conference', 'page_start': '', 'page_end': ''}

Thought: The paper is published on AAAI Spring Symposium - MLPS.

Action: Finish[AAAI Spring Symposium - MLPS]
<Solution complete>"""]


COTQA_SIMPLE6 = """Question: How many extra minutes did the DL1575 flight take from ATL to MCO on 2022-01-12?
Let's think step by step. I need to first load the flights database with LoadDB[flights]. And then, I need to filter the information I want according to the query with FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]. Then, I need to get the value of the departure time from the database with GetValue[DepDelay]. We then need to know the arrival delayed time with GetValue[ArrDelay]. To compute the extra minutes, we need to subtract the departure delayed time from the arrival delayed time with Calculate[(-17)-(-7)].  After calculation, we know that the answer is -10.
Action 1: LoadDB[flights]-->FilterDB[Flight_Number_Marketing_Airline=1575, FlightDate=2022-01-12, Origin=ATL, Dest=MCO]-->GetValue[DepDelay]-->GetValue[ArrDelay]-->Calculate[(-17)-(-7)]-->Finish[-10]
"""
COT = """
"""

COT_REFLECT = """
"""

REFLECTIONS = """
"""