examples = [

## How many events happen on 2022/03/25 in the agenda table?
# 

"""Question: How many events happen on 2022/03/25 in the agenda table?\n\nThought: To find out how many events happened on a specific date, I will need to load the database that contains the agenda information and then filter the database based on the specific date.\n\nAction: LoadDB\n\nAction Input: {"DBName": "agenda"}\n\nObservation: We have successfully loaded the agenda database, including the following columns: event, start_time, end_time, location, person, date, id.\n\nThought: Now that the "agenda" database is loaded, the next step is to filter this database to find entries that match the specific date, 2022/03/25.\n\nAction: FilterDB\n\nAction Input: {"condition": "date=2022/03/25"}\n\nObservation: We have successfully filtered the data (31 rows).\n\nThought: There are 31 events happening on 2022/03/25 in the agenda table.\n\nAction: Finish\n\nAction Input: {"answer": "31"}\n\nObservation: Answer is CORRECT.""",


## How many people are unavailble between 10:30 AM and 12:00 PM on 2022/08/09 in the agenda table?
# 10

"""Question: How many people are unavailble between 10:30 AM and 12:00 PM on 2022/08/09 in the agenda table?\n\nThought: To find out how many people are unavailable during a specific time frame on a specific day, I need to access the agenda database. Then I'll filter the database to find entries that overlap with the time interval on 2022/08/09 between 10:30 AM and 12:00 PM.\n\nAction: LoadDB\n\nAction Input: {"DBName": "agenda"}\n\nObservation: We have successfully loaded the agenda database, including the following columns: event, start_time, end_time, location, person, date, id.\n\nThought: Now that the "agenda" database is loaded, I must filter this database to find entries for events that take place on 2022/08/09 during the specified time frame. This involves identifying entries where either the start time is before or at 12:00 PM and the end time is after or at 10:30 AM on the specified date.\n\nAction: FilterDB\n\nAction Input: {"condition": "date=2022/08/09, start_time<=12:00, end_time>=10:30"}\n\nObservation: We have successfully filtered the data (12 rows).\n\nThought: Given these events, I need to count the unique individuals involved in these events to understand how many people are unavailable.\n\nAction: GetValue\n\nAction Input: {"column_name": "person"}\n\nObservation: {"person": ["Daniel", "Charlotte", "Adrian", "Maisie", "Poppy", "Jade", "Brian", "Daniel", "Jessica", "Zara", "Andrew", "Charlotte"]}\n\nThought: The observation reveals that there are multiple names listed, including duplicates. To ascertain the exact number of unique individuals who are unavailable, we must remove duplicate names from the list.\n\nAction: PythonInterpreter\n\nAction Input: {"Python": "names = ['Daniel', 'Charlotte', 'Adrian', 'Maisie', 'Poppy', 'Jade', 'Brian', 'Daniel', 'Jessica', 'Zara', 'Andrew', 'Charlotte']\n\nunique_names = set(names)\n\nlen(unique_names)"}\n\nObservation: 10\n\nThought: There are 10 unique people who are unavailable between 10:30 AM and 12:00 PM on 2022/08/09 in the agenda table.\n\nAction: Finish\n\nAction Input: {"answer": "10"}\n\nObservation: Answer is CORRECT.""",


## When should I schedule a meeting with Poppy from 9:00 AM to 6:00 PM on 2022/04/08 in the agenda table?
# 9:00 AM-6:00 PM #  9:00 AM-8:00 PM, 10:00 PM-6:00 PM

"""Question: When should I schedule a meeting with Poppy from 9:00 AM to 6:00 PM on 2022/04/08 in the agenda table?\n\nThought: To find an appropriate time slot to schedule a meeting with Poppy from 9:00 AM to 6:00 PM on 2022/04/08, I need to access the agenda database and filter for entries involving Poppy on that date. Then, I'll identify the time slots where Poppy is busy to determine when she is available.\n\nAction: LoadDB\n\nAction Input: {"DBName": "agenda"}\n\nObservation: We have successfully loaded the agenda database, including the following columns: event, start_time, end_time, location, person, date, id.\n\nThought: With the agenda database loaded, the next step is to filter the database for entries that involve Poppy on the specific date, 2022/04/08, to see her commitments for that day.\n\nAction: FilterDB\n\nAction Input: {"condition": "person=Poppy, date=2022/04/08"}\n\nObservation: We have successfully filtered the data (1 rows).\n\nThought: With only one entry for Poppy on 2022/04/08, it suggests she has a single commitment that day in the agenda database. To schedule a meeting, I need to know the time of this commitment to identify available slots before and after this event within the 9:00 AM to 6:00 PM time frame.\n\nAction: GetValue\n\nAction Input: {"column_name": "start_time, end_time"}\n\nObservation: {"start_time": "20:00:00", "end_time": "22:00:00"}\n\nThought: The commitment for Poppy on 2022/04/08 is from 8:00 PM to 10:00 PM, which falls outside the specified window of 9:00 AM to 6:00 PM. Given this information, Poppy's entire availability spans the requested timeframe since her commitments begin after 6:00 PM.\n\nAction: Finish\n\nAction Input: {"answer": "9:00 AM-6:00 PM"}\n\nObservation: Answer is CORRECT.""",


## What events does Georgina have on 2022/04/28 in the agenda table?
# Sightseeing Tour

"""Question: What events does Georgina have on 2022/04/28 in the agenda table?\n\nThought: To find the events Georgina has on 2022/04/28, I need to load the agenda database and specifically filter for entries involving Georgina on that date. This will help isolate her commitments for the day.\n\nAction: LoadDB\n\nAction Input: {"DBName": "agenda"}\n\nObservation: We have successfully loaded the agenda database, including the following columns: event, start_time, end_time, location, person, date, id.\n\nThought: Now, with access to the agenda database, I will filter for entries that match Georgina's engagements on 2022/04/28 specifically.\n\nAction: FilterDB\n\nAction Input: {"condition": "person=Georgina, date=2022/04/28"}\n\nObservation: We have successfully filtered the data (1 rows).\n\nThought: Since there is one row returned for Georgina's engagements on 2022/04/28, I need to retrieve the details of the event(s) she has on that day.\n\nAction: GetValue\n\nAction Input: {"column_name": "event, start_time, end_time, location"}\n\nObservation: {"event": "Sightseeing Tour", "start_time": "09:00:00", "end_time": "12:00:00", "location": "Statue of Liberty"}\n\nThought: Georgina has one event on 2022/04/28, which is a "Sightseeing Tour" scheduled from 9:00 AM to 12:00 PM at the "Statue of Liberty".\n\nAction: Finish\n\nAction Input:  {"answer": "Sightseeing Tour"}\n\nObservation: Answer is CORRECT.""",


## How many dates in the agenda table have Stephen scheduled?
# 86

"""Question: How many dates in the agenda table have Stephen scheduled?\n\nThought: To figure out how many dates have Stephen scheduled in the agenda table, I should first load the agenda database. Then, I'll filter the database to get entries related to Stephen. After that, I'll extract the dates from these entries and count how many unique dates there are.\n\nAction: LoadDB\n\nAction Input: {"DBName": "agenda"}\n\nObservation: We have successfully loaded the agenda database, including the following columns: event, start_time, end_time, location, person, date, id.\n\nThought: Now the agenda database is loaded, the next step is to filter for entries that involve Stephen specifically.\n\nAction: FilterDB\n\nAction Input: {"condition": "person=Stephen"}\n\nObservation: We have successfully filtered the data (97 rows).\n\nThought: Considering the entries have been filtered, the next practical step is to extract all the dates from these entries to determine how many unique dates there are.\n\nAction: GetValue\n\nAction Input: {"column_name": "date"}\n\nObservation: {"date": ["2022/01/29", "2022/12/18", "2022/12/29", "2022/01/24", "2022/05/16", "2022/06/23", "2022/03/20", "2022/01/29", "2022/02/15", "2022/03/30", "2022/02/06", "2022/06/17", "2022/10/04", "2022/01/22", "2022/06/04", "2022/07/12", "2022/02/29", "2022/05/25", "2022/09/10", "2022/11/04", "2022/04/12", "2022/04/19", "2022/08/10", "2022/11/28", "2022/07/26", "2022/03/18", "2022/08/08", "2022/02/17", "2022/02/25", "2022/11/20", "2022/10/22", "2022/04/30", "2022/08/26", "2022/12/14", "2022/11/30", "2022/07/05", "2022/03/20", "2022/04/01", "2022/05/30", "2022/08/06", "2022/11/05", "2022/12/29", "2022/12/30", "2022/04/01", "2022/12/19", "2022/07/18", "2022/07/01", "2022/11/24", "2022/04/28", "2022/05/19", "2022/10/03", "2022/02/28", "2022/05/13", "2022/04/05", "2022/04/13", "2022/01/28", "2022/05/22", "2022/06/17", "2022/06/26", "2022/06/08", "2022/09/28", "2022/01/06", "2022/09/17", "2022/08/27", "2022/01/06", "2022/08/08", "2022/01/03", "2022/02/23", "2022/11/18", "2022/01/01", "2022/11/17", "2022/02/26", "2022/07/06", "2022/04/09", "2022/05/23", "2022/12/20", "2022/03/13", "2022/01/27", "2022/05/23", "2022/04/16", "2022/01/14", "2022/03/29", "2022/02/24", "2022/08/28", "2022/02/04", "2022/12/11", "2022/04/04", "2022/12/08", "2022/07/19", "2022/06/04", "2022/02/25", "2022/12/28", "2022/10/19", "2022/08/13", "2022/03/06", "2022/05/23", "2022/11/29"]}\n\nThought: With the dates retrieved, I need to count the number of unique dates to answer the question.\n\nAction: PythonInterpreter\n\nAction Input: {"Python": "dates = ['2022/01/29', '2022/12/18', '2022/12/29', '2022/01/24', '2022/05/16', '2022/06/23', '2022/03/20', '2022/01/29', '2022/02/15', '2022/03/30', '2022/02/06', '2022/06/17', '2022/10/04', '2022/01/22', '2022/06/04', '2022/07/12', '2022/02/29', '2022/05/25', '2022/09/10', '2022/11/04', '2022/04/12', '2022/04/19', '2022/08/10', '2022/11/28', '2022/07/26', '2022/03/18', '2022/08/08', '2022/02/17', '2022/02/25', '2022/11/20', '2022/10/22', '2022/04/30', '2022/08/26', '2022/12/14', '2022/11/30', '2022/07/05', '2022/03/20', '2022/04/01', '2022/05/30', '2022/08/06', '2022/11/05', '2022/12/29', '2022/12/30', '2022/04/01', '2022/12/19', '2022/07/18', '2022/07/01', '2022/11/24', '2022/04/28', '2022/05/19', '2022/10/03', '2022/02/28', '2022/05/13', '2022/04/05', '2022/04/13', '2022/01/28', '2022/05/22', '2022/06/17', '2022/06/26', '2022/06/08', '2022/09/28', '2022/01/06', '2022/09/17', '2022/08/27', '2022/01/06', '2022/08/08', '2022/01/03', '2022/02/23', '2022/11/18', '2022/01/01', '2022/11/17', '2022/02/26', '2022/07/06', '2022/04/09', '2022/05/23', '2022/12/20', '2022/03/13', '2022/01/27', '2022/05/23', '2022/04/16', '2022/01/14', '2022/03/29', '2022/02/24', '2022/08/28', '2022/02/04', '2022/12/11', '2022/04/04', '2022/12/08', '2022/07/19', '2022/06/04', '2022/02/25', '2022/12/28', '2022/10/19', '2022/08/13', '2022/03/06', '2022/05/23', '2022/11/29']\nunique_dates = set(dates)\nlen(unique_dates)"}\n\nObservation: 86\n\nThought: Stephen has events scheduled on 86 unique dates in the agenda table.\n\nAction: Finish\n\nAction Input:  {"answer": "86"}\n\nObservation: Answer is CORRECT.""",

]