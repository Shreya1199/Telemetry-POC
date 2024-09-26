#!/usr/bin/env python
# coding: utf-8

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
import os
import pandas as pd
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import pandasql as ps
import json
import re
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)


genai.configure(api_key='')
model = genai.GenerativeModel('gemini-1.5-flash')
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")


 
def get_text_chunks(text):
    try:
        chunks = []
        num_rows = len(df)
        chunk_size = 1
        
        for start_row in range(0, num_rows, chunk_size):
            end_row = min(start_row + chunk_size, num_rows)
            chunk = ""
            for index in range(start_row, end_row):
                row = df.iloc[index]
                for col in df.columns:
                    chunk += f"{col}: {row[col]}\n"
                chunk += "\n"
            chunks.append(chunk)
        
        return chunks
        
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    



#conversion of embeddings

#conversion of embeddings
try:
    file_path = r"Dummy_dataset.csv"       #replace the file_path for using in different source
    df = pd.read_csv(file_path)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # chunk = get_text_chunks(df)
    # new_db = FAISS.from_texts(chunk, embedding=embeddings)
    # new_db.save_local("faiss_index")
    
    #for category 2
    sql_df = df.copy()
    sql_df["refer_index"] = sql_df.index
    df_copy = sql_df.copy()
    df_copy = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    df_copy = df_copy.applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
    df_copy["refer_index"] = df_copy.index
    
    #conversion of embeddings for temporary dataset
    # chunk_temp = get_text_chunks(df_copy)
    # new_db_temp = FAISS.from_texts(chunk_temp, embedding=embeddings)
    # new_db_temp.save_local("faiss_index_temp")
    
except Exception as e:
        st.write(f"An unexpected error occurred: {str(e)}")


#this is the conversational chain for getting the telemetry using scenario ( Category 1 )
def get_conversational_chain_telemetry(df):
    try:
        context = ""
        for index, row in df.iterrows():
            context += f"Row {index+1}:\n"
            for col in df.columns:
                context += f"{col}: {row[col]}\n"
            context += "\n"

        prompt_template = """
    context:
    {context}
    question:
    {question}
    Context:
    The data (df) provided is the telemetry data of meeting feature in Microsoft teams across different platforms. 
    Available columns - 
    1) Platform - It indicates the platform of the action occured. Eg: IOS, Android, Maglev, Web
    2) WorkLoad - It indicates wheather it is a collab action or not
    3) SubWorkLoad - It denotes wheather the action done is a meeting action or not
    4) SubWorkLoadScenario - It denotes what kind of action that has been done within the meeting feature
    5) Description - This provides more details of the action that has been performed
    6) Action_Type - This denotes wheather the captured telemetry is primary or secondary
    8) action_gesture
    9) module_name
    10) module_type
    11) action_outcome
    12) panel_type
    13) action_scenario
    14) action_scenario_type
    15) action_subWorkLoad
    16) thread_type
    17) databag.rsvp
    18) module_summary
    19) target_thread_type
    20) databag.meeting_rsvp
    21) databag.is_toggle_on
    22) databag.community_rsvp
    23) main_entity_type
    24) main_slot_app
    25) eventInfo.identifier
    26) databag.action_type
    27) subNav_entity_type
    You need to work as a telemetry expert and give important telemetry markers by understanding the provided user question and generating a relevant
    SQL query
 
    Instructions:
    You need to create a sql query for fetching telemetry columns by understanding the question asked by the user
    Example Question: Give me the telemetry/markers for joining a meeting
    In the above question, "joining a meeting" is the action that user need the telemetry for.
    Whenever a user asks telemetry for a specific action, find the similar item in the SubWorkLoadScenario list, then use that item in the SQL query like SubWorkLoadScenario = 'similar SubWorkLoadScenario list item'
    Expected Output Format
    Return the SQL query as plain text
    IMPORTANT:
    Identify the columns which should be used for select clause and columns which should be used inside where clause.
    Using the {context} understand the column name that is close relevant to the {question}
        -Ignore special character, compare it to {context} and then identify the close relevant 
        -Ignore case sensitive, compare it to {context} and identify the close relevant 
    Using the {context} identify the close relevant values mentioned in the {question} to construct the sql query
        -Ignore special character, compare it to {context} and then identify the close relevant 
        -Ignore case sensitive, compare it to {context} and identify the close relevant 
    Always assume that the user want only the "Primary" values records in the Action_Type column
    Constraints:
    Only the sql query generated in a single line.
    Don't include additional information
    Don't include any characters above or below the sql squery
    Example SQL Query and User Question - 
    1)Give me the telemetry for joining a meeting through calendar 
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_gesture],[module_name],[module_type],[action_outcome],[panel_type],[action_scenario],[action_scenario_type],[action_subWorkLoad],[thread_type],[databag.rsvp],[module_summary],[target_thread_type],[databag.meeting_rsvp],[databag.is_toggle_on],[databag.community_rsvp],[main_entity_type],[main_slot_app],[eventInfo.identifier],[databag.action_type],[subNav_entity_type] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Action_Type = 'Primary'
    2)Provide me with the markers for adding a meeting to calendar in IOS and Android 
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_gesture],[module_name],[module_type],[action_outcome],[panel_type],[action_scenario],[action_scenario_type],[action_subWorkLoad],[thread_type],[databag.rsvp],[module_summary],[target_thread_type],[databag.meeting_rsvp],[databag.is_toggle_on],[databag.community_rsvp],[main_entity_type],[main_slot_app],[eventInfo.identifier],[databag.action_type],[subNav_entity_type] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Platform in ('IOS', 'Android') AND Action_Type = 'Primary'
    3)Could you give me the telemetry markers for turning off the video in a meeting in Maglev
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_gesture],[module_name],[module_type],[action_outcome],[panel_type],[action_scenario],[action_scenario_type],[action_subWorkLoad],[thread_type],[databag.rsvp],[module_summary],[target_thread_type],[databag.meeting_rsvp],[databag.is_toggle_on],[databag.community_rsvp],[main_entity_type],[main_slot_app],[eventInfo.identifier],[databag.action_type],[subNav_entity_type] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Platform in ('Maglev') AND Action_Type = 'Primary'
    Important:
    The above examples are just for your understanding, don't copy the exact same thing from the prompt template in your response
    Include a platform filter in the query, if any of the following platforms—iOS, Android, Web, and Maglev—have been mentioned specifically in the user's question; if not, include all four platforms in the query.
    Always use the filter Action_Type = "Primary"
    Change the filter condition in the Platform column based on the user requirement, If user mentioned IOS, then provide the where condition as platform in ('IOS'). If user mentioned Web, then provide the where condition as platform in ('Web').If user mentioned Android and Maglev, then provide the where condition as platform in ('Android', 'Maglev')
    Like the above important instruction modify the query based on the user requirement
 
    
If specifically mentioned android or maglev or ios or web, select the mentioned platform values in the where condition with platform column
for ex: if asked for Android - Query: Platform = 'Android'
for ex: if asked for Ios and Android - Query: Platform in ['Android', 'IOS']
 
If the user requests all telemetry for all SubWorkLoadScenario, provide all the available data, but ensure that only the rows marked as "primary" are included
Example SQL Query and User Question - 
1)Could you provide all the telemetry data for each sub-workload
SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_gesture],[module_name],[module_type],[action_outcome],[panel_type],[action_scenario],[action_scenario_type],[action_subWorkLoad],[thread_type],[databag.rsvp],[module_summary],[target_thread_type],[databag.meeting_rsvp],[databag.is_toggle_on],[databag.community_rsvp],[main_entity_type],[main_slot_app],[eventInfo.identifier],[databag.action_type],[subNav_entity_type] FROM df  AND Action_Type = 'Primary'
 
2)Could you give me the telemetry details for all the sub-workloads
SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_gesture],[module_name],[module_type],[action_outcome],[panel_type],[action_scenario],[action_scenario_type],[action_subWorkLoad],[thread_type],[databag.rsvp],[module_summary],[target_thread_type],[databag.meeting_rsvp],[databag.is_toggle_on],[databag.community_rsvp],[main_entity_type],[main_slot_app],[eventInfo.identifier],[databag.action_type],[subNav_entity_type] FROM df  AND Action_Type = 'Primary'
 
3)Would you mind sharing the telemetry insights for all sub-workloads
SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_gesture],[module_name],[module_type],[action_outcome],[panel_type],[action_scenario],[action_scenario_type],[action_subWorkLoad],[thread_type],[databag.rsvp],[module_summary],[target_thread_type],[databag.meeting_rsvp],[databag.is_toggle_on],[databag.community_rsvp],[main_entity_type],[main_slot_app],[eventInfo.identifier],[databag.action_type],[subNav_entity_type] FROM df  AND Action_Type = 'Primary'

Take above examples as examples and generate query based on the users question
        
        """
     
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
     
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
     
        return chain,context
    
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

#this is a function to get response for getting the telemetry using scenario ( Category 1 )

def telemetry_response(user_question):
    try:
        chain,context = get_conversational_chain_telemetry(df)
        
        #reading the vector file
        vector_file = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_file.similarity_search(user_question)
        response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
        query = response["output_text"]
        if query.startswith('```sql'):
            query = query[6:]  # Remove the first 5 characters (```sql)
        if query.endswith('```'):
            query = query[:-3]  # Remove the last 3 characters (```)
        query = query.strip() 
        output_df = ps.sqldf(query)
        print(query)
        new_df= pd.DataFrame(output_df)
        if new_df.empty:
            return "Please try again with a different question, as I couldn’t find what you're looking for."
        # Convert to dictionary, removing only None values
        else:
            inter_res = [
                {k: v for k, v in row.items() if v is not None}
                for row in new_df.to_dict(orient='records')
            ]
            nan = float('nan')
            cleaned_list = [
                {k: v for k, v in d.items() if v == v} for d in inter_res]
            result = []  # To store the formatted output
         
            for item in cleaned_list:
                # Remove Action_Type if present
                item.pop("Action_Type", None)
                platform = item.pop("Platform")
                sub_workload_scenario = item.pop("SubWorkLoadScenario")
                # Create a string for Platform and SubWorkLoadScenario
                entry = f"On the platform {platform}, the telemetry generated for the action: {sub_workload_scenario}\n"
                # Convert the rest of the dictionary into a JSON-formatted string
                json_data = json.dumps(item, indent=4)
                # Remove curly braces {} from the JSON string
                json_data = json_data.replace("{", "").replace("}", "")
                # Append the combined string to the result list
                result.append(entry +"\n" + json_data )
         
            # Join the list into a single string
            final_result = '\n\n'.join(result)          
         
            # formatted_output = json.dumps(result, indent=2) # to format the output in a json format
            return final_result
        # formatted_output = json.dumps(result, indent=2) # to format the output in a json format
    except Exception as e:
        return "Sorry, couldn't process your question. Try again"


#this is a conversational chain for getting the scenario using marker ( Category 2 )

def get_conversational_chain_scenario(df):
    try:
        context = ""
        for index, row in df.iterrows():
            context += f"Row {index+1}:\n"
            for col in df.columns:
                context += f"{col}: {row[col]}\n"
            context += "\n"
       

        prompt_template = """
    
        context:
        {context}
    
        Question:
        {question}
    
        Important Consideration:
        1.Always strictly follow the below instructions, tasks, output and output format for any user question to generate the appropriate output.
        2.Don't include extra or additional information other than the listed.
        3.Look for columns, values and keywords only using the question, don't be creative and don't use other values,columns and keywords that is not close relevant to the value mentioned in question.
        4.If action or scenario is asked always provide SubWorkLoadScenario value don't give action_scenario value in any case of the question.
        5.Only if "action scenario" is mentioned as a combined term and consider action_scenario in any case of the question
        6.Always use where clause incase user asks for specific platform values or any platform values.
        7.Always include SQL functions such as COUNT or UNIQUE in the query if the question involves counting or identifying distinct values
        8.Only give accurate result according to the question.
        
        Instructions for Generating SQL Query:
        1. Identify the user's request based on the keywords in the question.
        
        2. Handling "action" and "action scenario" in the question:
            -If both "action" and "action scenario" are mentioned in the question:
                -Consider one "action" on its own, which is not immediately followed by "scenario," as referring to the SubWorkLoadScenario column and use it for the output.
                Example:
                -User Question: "Can you provide the action where action scenario is storeNewMeeting is used?"
                SQL Query: SELECT platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'storenewmeeting';
                -User Question: "can you provide me action where action scenario is store New Meeting is used in any platform"
                SQL Query: SELECT platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'storenewmeeting';
                
            
        3. Handling both "scenario" and "action scenario" in the question:
            -If the user mentions both "scenario" and "action scenario" in any part of the question, treat "scenario" alone (without being immediately preceeded by "action") as referring to the SubWorkLoadScenario column.
            -If "action" is followed by "scenario," treat it as referring to the action_scenario column.
             Example:
             - User Question: "Can you provide the scenario where action scenario is store NEw Meeting is used?"
             - SQL Query: `SELECT platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'storenewmeeting';
     
        4. Handling only "action" in the question:
            -If the user mentions only "action" in the question, treat "action" alone (without being immediately followed by "scenario") as referring to the SubWorkLoadScenario column.
             Example:
             - User Question: "can you provide me the action where automatic is used?"
             - SQL Query: `SELECT platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automatic';
     
        5. Handling only "scenario" in the question:
            -If the user mentions only "scenario" in the question, treat "scenario" alone (without being immediately preceeded by "action") as referring to the SubWorkLoadScenario column.
             Example:
             - User Question: "can you provide me the scenario where is automatic used?"
             - SQL Query: `SELECT platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automatic';
     
        6. Handling both "action" and "action gesture" in the {question}:
            -If the user mentions both "action" and "action scenario" in any part of the question, treat "action" alone (without being immediately followed by "scenario") as referring to the SubWorkLoadScenario column.
            -If "action" is followed by "gesture" treat it as referring to the action_gesture column.
             Example:
             - User Question: "can you provide me the action where action gesture is automatic"
             - SQL Query: `SELECT platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automatic';
     
        7. Handling both "scenario" and "action gesture" in the {question}:
            -If the user mentions both "scenario" and "action gesture" in any part of the question, treat "scenario" alone (without being immediately preceeded by "action") as referring to the SubWorkLoadScenario column.
            -If "action" is followed by "gesture," treat it as referring to the action_gesture column.
             Example:
             - User Question: "can you provide me the scenario where action gesture is automatic"
             - SQL Query: `SELECT platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automatic';
     
        8. Handling both "action scenario" and "action scenario" in the question:
            -If the user mentions "action scenario" as a combined term in any part of the question more than one time, refer it as action_scenario column
             Example:
             - User Question: "Can you provide the action scenario where action scenario is enter Meeting is used?"
             - SQL Query: `SELECT platform, action_scenario FROM df_copy WHERE action_scenario = 'entermeeting'; 
    
        9. Handling only "action scenario" in the question:
            -If the user mentions "action scenario" as a combined term in any part of the question one time, refer it as action_scenario column
             Example:
             - User Question: "can you provide me count of unique scenario for each platform"
             - SQL Query: `SELECT platform, COUNT(DISTINCT(action_scenario)) AS "count_unique_action_scenario" FROM df_copy GROUP BY platform; 
        
        10. When multiple columns are requested:
           - If the question combines both standalone terms and phrases, ensure to include all requested columns and apply the correct filtering logic.
             Example:
             - User Question: "Can you provide the action and module name where in action scenario where api is used?"
             - SQL Query: `SELECT platform, SubWorkLoadScenario, module_name FROM df_copy WHERE action_scenario LIKE '%api%';
             - User Question: "Can you provide the scenario and module name where in action scenario where API is used?"
             - SQL Query: `SELECT platform, SubWorkLoadScenario, module_name FROM df_copy WHERE action_scenario LIKE '%api%';
             
        11. Handling more than 1 word value as combined values:
            - If the value has more than 1 word, leave no space between words.
            Example: 
                -if user question contains "store new meeting" consider it as "storenewmeeting"
                -if user question contains "Update meeting" consider it as "updatemeeting"
                -if user question contains "modify stage View" consider it as "modifystageview"
    
        12. Handling uppercase characters:
            - If there is uppercase characters existing in values, convert them to lowercase characters in any case of the question.
            Example: 
                -if user question contains "Store New Meeting" consider it as "storenewmeeting"
                -if user question contains "Update meeting" consider it as "updatemeeting"
                -if user question contains "modify STAGE View" consider it as "modifystageview"
         
        Tasks:
        1. Construct an SQL query that selects the relevant column (e.g., `SubWorkLoadScenario`,platform,action_scenario) from the table `df_copy` based on the identified column and value.
        2. Identify the columns which should be used for select clause and columns which should be used inside where clause.
        3. Using the {context} understand the column name that is close relevant to the {question}
            -Ignore special character, compare it to {context} and then identify the close relevant 
            -Ignore case sensitive, compare it to {context} and identify the close relevant 
        4. Using the {context} identify the close relevant values mentioned in the {question} to construct the sql query
            -Ignore special character, compare it to {context} and then identify the close relevant 
            -Ignore case sensitive, compare it to {context} and identify the close relevant 
        5. Always include "platform" column as mandatory for any outputs with asked columns in any case of the question  
        6. Always include "refer_index" column as mandatory for any outputs with asked columns in any case of the question  
    
        Output:
        1. An SQL query that matches the user's request with platform column in the output columns if it is not in the query generated.
        2. The query should follow the instructions mentioned above strictly.
        2. Only give what the user has asked and don't include anything extra for the query
    
        Output Format:
        1.Only the sql query generated in a single line.
        2.Don't include additional information like "sql" word or any special character before or after query
        3.Don't include any characters above or below the sql squery
        4.Strictly follow only this below format and no other should be included extra to it.
        Example format:
            1. SELECT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'screenSharing';
            2. SELECT refer_index, SubWorkLoadScenario,platform FROM df_copy WHERE action_scenario = 'storenewmeeting' 
            3. SELECT refer_index, SubWorkLoadScenario,platform FROM df_copy WHERE action_gesture = 'touch' AND module_name = 'savenewsession' AND module_type = 'pushbutton' AND action_outcome = 'send' AND platform = 'android';
            
    
        Example:
        User Question: "Can you provide me the scenario where action scenario is entermeeting is used?"
        SQL Query should be like this: SELECT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'entermeeting';
    
        User Question: "Can you provide me the action where action scenario is entermeeting is used?"
        SQL Query should be like this: SELECT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'entermeeting';
        
        User Question: "can you provide me the scenario where action gesture is automated, module name is API workflow, module type is planner in android"
        SQL Query should be like this: SELECT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automated' AND module_name = 'apiworkflow' AND module_type = 'planner' AND platform = 'android';
    
        User Question: "Can you provide the action where action scenario is storenewmeeting is used?"
        SQL Query should be like this: SELECT refer_index, SubWorkLoadScenario,platform FROM df_copy WHERE action_scenario = 'storenewmeeting' 
    
        User Question: "can you provide me the action where action gesture is automated"
        SQL Query should be like this: `SELECT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automated';
    
        Addtional considerations for generating qunatification query:
    
        1. If the user asks for unique for certain column, the query should look like this:
            Example:
            prompt: provide me all distinct action scenarios 
            query: Select DISTINCT(action_scenario) as "distinct_action_scenario" from df_copy
        
        2. If the user asks for count of unique for certain column, the query should look like this:
            Example:
            prompt: provide me all count of distinct action scenarios 
            query: Select COUNT(DISTINCT(action_scenario)) as "count_distinct_action_scenario" from df_copy
    
        3. If the user asks for count for certain column and group by certain column, the query should look like this:
            Example:
            prompt: provide me all distinct action scenario for each platform
            query: Select platform, COUNT(action_scenario) as "count_action_scenario" from df_copy and group by platform
    
        4. If the user asks for count for certain column and group by certain column for a certain value, the query should look like this:
            Example:
            prompt: can you provide me count of action gesture is tap for each platform
            query: Select platform, COUNT(action_gesture) as "count_action_gesture" from df_copy and group by platform and action_gesture = "touch"
    
    
        """
     
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
     
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
     
        return chain,context
    
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"    

#this is a function to get response for getting the scenario using marker ( Category 2 )

def scenario_response(user_question):
    try:
        chain,context = get_conversational_chain_scenario(df_copy)
        
        #reading the vector file
        vector_file = FAISS.load_local("faiss_index_temp", embeddings, allow_dangerous_deserialization=True)
        docs = vector_file.similarity_search(user_question)
        response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
        
        print(response["output_text"])
        query = response["output_text"]
        if query.startswith('```sql'):
            query = query[6:]  # Remove the first 5 characters (```sql)
        if query.endswith('```'):
            query = query[:-3]  # Remove the last 3 characters (```)
        query = query.strip() 
        inter_df = ps.sqldf(query)
        if inter_df.empty:
            return "Please try again with a different question, as I couldn’t find what you're looking for."
        else:
            common_columns = inter_df.columns.intersection(sql_df.columns)
            matching_index = inter_df["refer_index"].tolist()
            output_df = sql_df.loc[matching_index,common_columns]
            result_df = output_df.drop(columns = ["refer_index"])
            result_df = result_df.reset_index(drop = True)
            # result_df = result_df.style.hide(axis='index')
            
            return result_df

    except Exception as e:
        return "Sorry, couldn't process your question. Try again"
 


#this is the conversational chain for recommending new telemetry ( Category 3)
 
def get_Recommendation(df):
    try:
        
        context = ""
        for index, row in df.iterrows():
            context += f"Row {index+1}:\n"
            for col in df.columns:
                context += f"{col}: {row[col]}\n"
            context += "\n"

        prompt_template = """
       
        context:
        {context}
         
        Question:
        {question}
        Context:
     
        The provided data is telemetry data for Microsoft Teams meeting features.
        The data includes four platforms: iOS, Android, Maglev, and Web.
        Users have performed various actions in Microsoft Teams, and the corresponding telemetry has been captured.
        Task:
     
        ** Important ** : Here you need to give the recommendation of the new telemetry values based on those relevant actions in a dynamic Telemetry manner.
         
        First categorize the user queries for your understanding to provide the relevant  telemetry all column for that query
       
        if the user prompt action is related to the
     
        --Joining the meeting/Meeting - Joining  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
     
        Meeting - Joining through calender
        Meeting - Joining through chat
        Meeting - Joining through invite link
        Meeting - Joining through meeting id
     
        --rsvp/Meeting - RSVP this is the category name you need to bucket where action or subworkloadscenario related to the list below.
     
        Meeting - RSVP - Accept
        Meeting - RSVP - Decline
        Meeting - RSVP - Tentative
     
       
        --rsvp community or Community event - RSVP this is the category name you need to bucket where action or subworkloadscenario related to the list below.
     
        Community event - RSVP - Going in
        Community event - RSVP - May be
        Community event - RSVP - Can't go
     
        --leave a meet/Meeting Leave this is the category name you need to bucket where action or subworkloadscenario related to the list below.
     
        Meeting - Leave a meeting
     
        --leave by end for all meet / Meeting leave by all this is the category name you need to bucket where action or subworkloadscenario related to the list below.
     
        Meeting - Leave a meeting by End for all
     
        --leave a community event/ Community event leave this is the category name you need to bucket where action or subworkloadscenario related to the list below.
     
        Community event - Leave a community event
     
        --Community event - Leavea community event  by End for all / Leave community event for all   this is the category name you need to bucket where action or subworkloadscenario related to the list below.
     
     
        Community event - Leavea community event  by End for all
     
        --Meeting reaction this is the category name you need to bucket where action or subworkloadscenario related to the list below.
     
        Meeting - reaction - Love
        Meeting - reaction - Like
        Meeting - reaction - Applause
        Meeting - reaction - Surprise
        Meeting - reaction - Laugh
        Meeting - reaction - Raise Hand
     
        joining the meeting  this is the categorize name you need to bucket where listed down
        Meeting - Joining through calender
        Meeting - Joining through chat
        Meeting - Joining through invite link
        Meeting - Joining through meeting id
        Meeting - Join button - Joining through calendar
        Meeting - Join button - Joining through chat
        Meeting - Join button - Joining through invite link
     
       
        --Opening the recations menu in the meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
            Meeting - Options - Open reaction menu
        --Inviting people inside the meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
           Meeting - Inviting people inside a meeting
        --Turning off the participants incoming video in the meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
           Meeting - Options - Turn off incoming video
        --Turning off the audio while joining the meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Audio off
        --Turning on the speaker while joining the meeting -  this is the categorize name you need to bucket   this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Speaker on iphone
          Meeting - Options - Speaker on
        --Resuming the call  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Resume a call
        --Put the call on hold  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Hold the call
        --Show live captions in the meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Live Caption
        --Customize the background in the meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Selecting a custom video background
        --share the content in the meeting  -this is the categorize name you need to bucket this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Presenting/Sharing photo
          Meeting - Options - Presenting/Sharing video
        --Toggling the all day option - this is the categorize name you need to bucket  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
           Meeting - Options - Toggle off - All day
           Meeting - Options - Toggle on - All day
        --Toggling the Online -  this is the categorize name you need to bucket  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Toggle on - Online
          Meeting - Options - Toggle off - Online
        --Change the layout - this is the categorize name you need to bucket  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Changing layout - Paginated view
          Meeting - Options - Changing layout - Gallery view
          Meeting - Options - Changing layout - Together view
        --Join the community event - this is the categorize name you need to bucket
        Community event - Join button - Joining through invite link
        Community event - Join button - Joining through post
        Community event - Join button - Joining through chat
        Community event - Join button - Joining through calendar
        Community event - Join button - Joining through events tab
        Community event - Joining through calender
        Community event - Joining through post
        Community event - Joining through invite link
        Community event - Joining through chat
        Community event - Joining through activity
        Community event - Joining through meeting tab
         
         
        --Sharing the screen in the meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Options - Presenting/Sharing screen
        --Turn on/off video - this is the categorize name that you need to bucket
           Meeting - Options - Turning on video
           Meeting - Options - Turning off video
        --Mute/Unmute the audio - this is the categorize name that you need to bucket
          Meeting - Options - Mute
          Meeting - Options - Unmute
        --Chat bubbles - this is the categorize name that you need to bucket
        Meeting - Options - Show chat bubbles
        Meeting - Options - Don't show chat bubbles
        --Add meeting to calendar  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Adding a meeting to calendar
        --add community event to calendar  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Community event - Adding a community event to calendar
        --Create a community event  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Community event - Creating a community event through calendar
          Community event - Create a community event
        -- cancel a community event  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
           Community event - Cancel a community event
        -- edit a community event  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Community event - Edit a community event
        -- create a meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Create a Meeting
        -- cancel a meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Cancel a Meeting
        -- edit a meeting  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
          Meeting - Edit a Meeting
    
         Note: Even if the user didn't mention about the category above listed in the question, understand the question and bucket that into the respective category
     
    **Second, if the user gives the prompt of the existing feature analysis and understands it, bucket the action in the respective category and go inside the respective category, and inside that bucket, see related actions or subworkload scenarios.
    and according to that, first understand the pattern of those existing features and give the new telemetries for the new feature the user needed.**
     
        If the user Query is related  to all platform then you need to fetch for all the platform that are in the data set (IOS,Maglev,Web,Android)
        where these platform data will be in the Platform column  you need to give all the platform related telemetries
       
        Example User Query
       
        "Recommend telemetry for the 'Angry Reaction' feature which is under the Reaction Menu"
       
        response template
        platform:Maglev
       
        New Feature-Angry Reaction
     
        **Based on the user query refer the relevent platform and provide the relvent of telemetry colums for that with the recommended values of you**
       
       
       
        platform:Ios
       
        New Feature-Angry Reaction
     
       **Based on the user query refer the relevent platform and provide the  colums for that with the recommended values of you**
       
        platform:Andriod
       
        New Feature-Angry Reaction
     
         **Based on the user query refer the relevent platform and provide the  colums for that with the recommended values of you**
     
        platform:Web
       
        New Feature-Angry Reaction
     
        **Based on the user query refer the relevent platform and provide the colums for that with the recommended values of you**
     
        **If the user's query pertains to a specific platform, respond with information relevant only to that platform**
       
        New Feature-Angry Reaction
           
        Example User Query:
       
        "Recommend telemetry for the 'Angry Reaction' feature which is under the Reaction Menu for Maglev platform"
     
        response template
       
        PlatForm:which platform the user query is asking
        New Feature-Angry Reaction
        **Refer to the repective all  column from action_gesture to subNav_entity_type   and give the recommended markers and its values**
       
       
        understand the pattern and give the recommended values it can be the random one as well but follow the formate as above for those column values
           
        Refer only the all marker columns from action_gesture to subNav_entity_type  to pull the relevent telemetry and its values
       
        for pulling the telemetry only refer to the telemetry from  column
        provide only from the primary values where Action the column name is Action Type
     
        Output format:
        Strictly follow the output format in any case of the question
        Require the output in key value pair starting from "action_gesture"
        Example Format:
     
        Platform: Platform name
        Scenario: Scenario name
        [
            key1: value1,
            key2: value2,
            keyn: valuen
        ]
     
        Note: the above example is for reference don't replicate the values of example for output
    
        Additional instructions:
        1. In the telemetry data try to include feature details in the data, as it is a new feature and should contain combination of the existing values and new values in it
        2. For every new feature the "action_scenario" should contain the detail regarding the new feature in it for any case of the question.
        3. Give all similar values of the bucketed feature for the new feature telemetries for any case of the question.
        4. Don't provide any example telemetries in the output only follow the output format and provide explanation and key points.
        5. Don't provide values with "nan" or as empty in any case of the question and don't include any explanation for the keys with "nan" or empty values.
        6. Provide explanation for all keys with values.
        7. Don't include Understand the user query and categorization in any part of the question.
        8. If the telemetry is for different platforms, in that case the telemetries values should be similar with the feature bucket of respective platform in any case of the question.
       
        """
     
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
     
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
     
        return chain,context

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
 
#this is a function to get response for Recommendation ( Category 3)
 
def Recommendation_response(user_question):
    try:
        chain,context = get_Recommendation(df)
       
        #reading the vector file
        vector_file = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_file.similarity_search(user_question)
        response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
     
        text = response["output_text"]
       
        json_sections = re.findall(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
       
        processed_sections = []
        last_end = 0
     
        for section in json_sections:
            # Extract text before this JSON section
            section_start = text.find(section, last_end)
            before_section = text[last_end:section_start].strip()
           
            # Clean and format the JSON part
            json_part = section.strip().replace('\n', '').replace(', ', ',').replace(' ', '')
            json_part = json_part.replace('[', '{').replace(']', '}')  # Handle [ and ] as well
            json_part = json_part.replace('}{', '},{')  # Handle cases where JSON sections are next to each other
           
            # Load the JSON part into a Python dictionary
            try:
                data = json.loads(json_part)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                data = {}
           
            # Filter out key-value pairs where the value is "nan"
            filtered_data = {key: value for key, value in data.items() if value != "nan"}
           
            # Convert the filtered dictionary back to a JSON string with indentation for readability
            filtered_string = json.dumps(filtered_data, indent=1)
            filtered_string = filtered_string.replace("{", "").replace("}", "")
            filtered_string = filtered_string.strip()
            filtered_string = " " + filtered_string
           
            # Extract text after this JSON section
            last_end = section_start + len(section)
            after_section = text[last_end:].strip()
           
            # Add the processed section
            processed_sections.append(f"{before_section}\n{filtered_string}\n")
           
            # Update last_end position
            last_end = last_end + len(after_section) - len(after_section.strip())
       
        # Add the remaining text after the last JSON section
        if last_end < len(text):
            processed_sections.append(text[last_end:].strip())
       
        result = ''.join(processed_sections)

     
       
        return result
        
    except Exception as e:
        return "Sorry, couldn't process your question. Try again" 
 


#this is the converstational chain for to categorize the prompt

def get_conversational_chain_prompt(user_question):
    
    try:
        prompt_template = """
        context:
        {context}
    
        Question:
        {question}
    
        **Instructions**: 
        Understand the user question and category the user prompt or question into one of the 4 categories -
            1. Telemetry: If the user prompts only for retrieval of telemetry or telemetry markers using any actions or scenarios categorize it into "Telemetry" category
            Example questions - Give me the telemetry markers for join a meeting action 
                              What is the telemetry for creating a community event in Maglev?
                              Can you provide telemetries for joining a meeting?
                              
            2. Subworkloadscenario: If the user prompts only for retrieval of any action  using any number of telemetry markers, categorize it into "Subworkloadscenario"
            Example questions - Can you provide me the action where savenewmeeting is used?
                              Can you provide me the action where module type is meetings and panel type is meeting join?
                              Can you provide the action where action acenario is meetingjoin in Web platform?
                              give me scenario where session join category is used?
                              
            3. Recommendation : If the user prompts are only related to suggest or recommend, categorize them into the "Recommendation" category
            Example questions - Give me the recommended telemetry or marker for the new feature where it is inside the meeting feature?
                                Provide the telemetry or marker for the new feature embedded within that feature(meeting)?
                                I need the telemetry or marker  for the new feature that's included inside that feature?
                                Please share the telemetry data for the new feature inside?
                                Suggest the telemetry inputs for the new feature inside the meeting?
                                what markers are available for the new features?
                                Provide the specific telemetries for the new features?
                                can you give me the telemetry for the angry reaction?
                                
            4. Others:
                Any user prompt that does not directly relate to the three categories above—telemetry, subworkload scenarios, or telemetry recommendations—should be categorized as "Others."
                
                Use the following criteria to categorize a question into "Others":
                
                    -General or unrelated queries, as well as broad questions about telemetry, the dataset, or any queries that do not specifically request telemetry markers, actions, or scenarios.
                    -Non-retrieval and non-recommendation questions related to telemetries.
                    -Gibberish text or random, nonsensical input.
                    -Non-literal inputs, such as symbols or numbers without meaningful context.
                    -Empty inputs, consisting of only spaces or blank entries.
                    
                If the user question matches any of these criteria, it should be strictly categorized as "Others" in any case of the question.
                Example questions:
                -What is telemetry
                -What is earth
                -What is subworkloadscenario

        Important notes:
        - All non retrieval questions and all non recommendation questions should be categorized under "others" in any case of the question.
        - The question which doesn't make any meaning or context also categorized under "others" in any case of the question.
        - The general question which is related to the dataset or telemetries should also be categorized under "others" in any case of the question.
        - Questions related to column names of the dataset should also be categorized as "others" in any case of the question.
        - Questions with no meaningfulness also be categorized as "others" in any case of the question.
        
        **Output Format**:
            - The output should be either "Telemetry" or "Subworkloadscenario" or "Recommendation" or "others" based on the user question
        
    
        Example User Query and Expected Response:
        - **User Query**: "Give me the data for joining a meeting through chat button."
        - **Expected Response**: 
            - Telemetry
        - **User Query**: "For what scenario I should use savenewmeeting as Action scenario."
        - **Expected Response**: 
            - Subworkloadscenario  
        - **User Query**: "For what action I should use tap as Action gesture."
        - **Expected Response**: 
            - Subworkloadscenario
        - **User Query**: "FI want the action associated with the following telemetry panel_type - stageSwitcher"
        - **Expected Response**: 
            - Subworkloadscenario
        - **User Query**: "Give me the recommended telemetry or marker for the new feature where it is inside the meeting feature."
        - **Expected Response**:
            -Recommendations
        - **User Query**: "I need the telemetry or marker  for the new feature that's included inside that feature."
        - **Expected Response**:
            -Recommendations
        - **User Query**: "what markers are available for the new features?"
        - **Expected Response**:
            -Recommendations
        - **User Query**: "can you provide me count of unique action scenario for each platform"
        - **Expected Response**:
            -Subworkloadscenario
        - **User Query**: "Give me scenario where tfl is used"
        - **Expected Response**:
            -Subworkloadscenario            
        - **User Query** : What is earth
        - **Expected Response**:
            -others
        - **User Query** : What is telemetry
        - **Expected Response**:
            -others 
        - **User Query** : What is subworkloadscenario
        - **Expected Response**:
            -others 
        - **User Query** : meant by telemetry
        - **Expected Response**:
            -others 
        - **User Query** : telemetry
        - **Expected Response**:
            -others 
    
        IMPORTANT : Give the output in the json format
        
        """
     
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
     
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
        return chain

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
    

#this is a function to categorize the prompt

def category_prompt(user_question):

    try:
        chain = get_conversational_chain_prompt(df)
        
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        
        if "Telemetry" in response["output_text"]:
            return "Telemetry"
        elif "Subworkloadscenario" in response["output_text"]:
            return "Subworkloadscenario"
        elif "Recommendation" in response["output_text"]:
            return "Recommendation"
        elif "others" in response["output_text"]:
            return "others"          
    except Exception as e:
        return "Sorry, couldn't process your question. Try aasking questions related to telemetry"



def user_input(user_question):
 
    global flag 
    flag = category_prompt(user_question)
    print(flag)
    
    if flag=="Telemetry":
        result = telemetry_response(user_question)
    elif flag=="Subworkloadscenario":
        result = scenario_response(user_question)
    elif flag=="Recommendation":
        result=Recommendation_response(user_question)
    elif flag=="others":
        result = "Please provide me questions related to telemetries or scenarios regarding meeting feature of teams, Sorry I couldn't process this."
        
        
    print(result)
        
    return result



def main():

#codes to get the get help/report a bug and an about button
    st.set_page_config(layout="wide", menu_items = {'Get Help': 'mailto:v-shreyashaw@microsoft.com,v-ajithis@microsoft.com,v-rohithp@microsoft.com,v-saransu@microsoft.com?cc=v-ishashiny@microsoft.com','Report a Bug':'mailto:v-shreyashaw@microsoft.com,v-ajithis@microsoft.com,v-rohithp@microsoft.com,v-saransu@microsoft.com?cc=v-ishashiny@microsoft.com','About':"This POC Telemetry Bot is made up test data of TFL Meeting Features. Using this chatbot, you can able to get the telemetry/markers for a particular action, get the action related to a particular telemetry, it can recommend you telemetries for a new introduced feature as well"} )

    hide_git = """
    <style>
    /* Hide the GitHub button */
    button[title="View on GitHub"] {
        display: none;
    }
    </style>
"""

# side bar configurations
    with st.sidebar:
        st.sidebar.markdown("<h1 style='font-size:20px; text-align:center;'>DESCRIPTION</h1>", unsafe_allow_html=True)
  
        st.markdown('<p style="font-size:14.5px; font-style: italic;">This Telemetry Chatbot has been fed with the telemetry data of meeting features of TFL and it can answer queries related to it. Please refer to the below section to understand its functionalities. </p>', unsafe_allow_html=True)   
        st.sidebar.markdown("<h1 style='font-size:20px; text-align:center;'>FUNCTIONALITY</h1>", unsafe_allow_html=True)
        st.sidebar.markdown(
        """
        <style>
        .main-point {
            font-size: 14px;  
            font-weight: bold; 
        }
        .sub-point {
            font-size: 14px;  
            font-style: italic; 
        }
        </style>
    
        <div class="main-point">1. Fetch Telemetry Markers for an Existing Action</div>
        <div class="sub-point">Provide necessary telemetry markers related to TFL Meeting features</div><br>
    
        <div class="main-point">2. Fetch Actions corresponding to Telemetry Markers</div>
        <div class="sub-point">Provide relevant actions corresponding to 'n' number of telemetry markers</div><br>
    
        <div class="main-point">3. Recommend Telemetry for New Features</div>
        <div class="sub-point">Recommend meaningful telemetries for new feature introductions</div>
        """,
        unsafe_allow_html=True)

#codes to remove the close button in sidebar        
    remove_cross_sidebar = """
    <style>
    [data-testid="baseButton-header"] {
        display: none;
    }
    </style>
    """
    
#codes to adjust the padding in sidebar        
    padding_sidebar_style = """
    <style>
    [data-testid="stSidebarUserContent"] {
        padding-top: 8%;
        padding-bottom: 10%;
    }
    </style>
    """
    
#codes to make the sidebar static
    sidebar_width_static = """
    <style>
            [data-testid="stSidebar"] {
                width: 17%; 
                min-width: 17%;
                max-width: 17%;
            }
            [data-testid="stSidebarResizer"] {
                display: none;
            }
    </style>
    """
    
#code to decrease the top padding
    padding_main = """
    <style>
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 0;
        padding-right: 0;
        padding-left: 0;
        width: 70%;
    }
    </style>
    """
    
    height_adjustment = """
    <style>
    .custom-height {
        height: 40px; 
        overflow: auto; 
        font-size: 25px; 
        font-weight: bold; 
    }
    </style>
    """

    styles1 = """
    <style>
    button[title="View fullscreen"] {
        display: none;
    }
    .stChatMessage.st-emotion-cache-janbn {
    flex-direction: row-reverse;
    text-align: right;
    max-width:70%;
    width: auto;
    height: auto;
    margin-left: auto;
    }
    .stChatMessage.st-emotion-cache-janbn0.eeusbqq4 {
    flex-direction: row-reverse;
    text-align: right;
    max-width:70%;
    width: auto;
    height: auto;
    margin-left: auto;
    }

    .st-emotion-cache-1ir3vnm.eeusbqq3 {
    justify-content: flex-start;
    writing-mode: horizontal-tb;
    </style>
    """

    styles2 = """
    <style>
    .stChatMessage.st-emotion-cache-1c7y2kd {
    flex-direction: row-reverse;
    text-align: right;
    max-width:70%;
    width: auto;
    height: auto;
    margin-left: auto;
    }
    .stChatMessage.st-emotion-cache-1c7y2kd.eeusbqq4 {
    flex-direction: row-reverse;
    text-align: right;
    max-width:70%;
    width: auto;
    height: auto;
    margin-left: auto;
    }

    .st-emotion-cache-1c7y2kd.eeusbqq4 {
    justify-content: flex-start;
    writing-mode: horizontal-tb;
    </style>
    """

    st.markdown(
    """
<style>
    [aria-label="Chat message from user"] div [data-testid=stVerticalBlock],
[aria-label="Chat message from user"] div [data-testid=stVerticalBlock]  * {
    width: auto !important;
}
</style>
    """, 
    unsafe_allow_html=True)

#calling all the above css 
    st.markdown(styles1, unsafe_allow_html=True)
    st.markdown(styles2, unsafe_allow_html=True)
    st.markdown(hide_git, unsafe_allow_html=True)
    st.markdown(remove_cross_sidebar, unsafe_allow_html=True)
    st.markdown(padding_sidebar_style, unsafe_allow_html=True)
    st.markdown(sidebar_width_static, unsafe_allow_html=True)
    st.markdown(padding_main, unsafe_allow_html=True)
    st.markdown(height_adjustment, unsafe_allow_html=True)

    try:
        
        # Create a container for logos and title with horizontal layout
        col1, col2, col3 = st.columns([1.5, 2.1, 0.3])
      
        # Display logo on the left
        with col1:
            st.image("musigma.png", width=35) # Adjust width as needed

        # Display title in the center
        with col2:
            # Streamlit app layout
            st.markdown("<div class='custom-height'>Telemetry Bot</div>", unsafe_allow_html=True)

        # Display logo on the right
        with col3:
            st.image("microsoft_PNG13.png", width=35)  # Align the logo to the right
        # st.title("Telemetry bot")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["type"] == "code":
                    st.code(message["content"])
                elif message["type"] == "dataframe":
                    st.dataframe(message["content"])
                else:
                    st.write(message["content"])
        
        # User input section (This should appear only once)
        if st.session_state.messages==[]:
            with st.chat_message("user"):
                st.write("Hi, I'm here to help you with telemetry related queries")
                
            with st.chat_message("assistant"):
                st.write("Here the response will be displayed")
                
        user_question = st.chat_input("What would you like to process?")
             
        if user_question:
            
            if user_question.isdigit():
                # Set the query flag to True
                st.session_state.has_query = True
                # Display user message
                with st.chat_message("user"):
                    st.write(user_question)
                # Append user's message to chat history
                st.session_state.messages.append({"role": "user", "type": "text", "content": user_question})
                response = "Please provide me questions related to telemetries or scenarios regarding meeting feature of teams, Sorry I couldn't process this."
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": response}) 

            elif re.fullmatch(r'\W+', user_question):
                # Set the query flag to True
                st.session_state.has_query = True
                # Display user message
                with st.chat_message("user"):
                    st.write(user_question)
                # Append user's message to chat history
                st.session_state.messages.append({"role": "user", "type": "text", "content": user_question})
                response = "Please provide me questions related to telemetries or scenarios regarding meeting feature of teams, Sorry I couldn't process this."
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})             


            else:
                # Set the query flag to True
                st.session_state.has_query = True
                
                # Process user input and output section
                output = user_input(user_question) 
                
                # Append user's message to chat history
                st.session_state.messages.append({"role": "user", "type": "text", "content": user_question})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(user_question)
            
                # Display output based on the flag and type
                if flag == "Subworkloadscenario":
                    with st.chat_message("assistant"):
                        # st.write("Here is the chatbot response:")
                        if isinstance(output, pd.DataFrame):
                            st.dataframe(output)
                            st.session_state.messages.append({"role": "assistant", "type": "dataframe", "content": output})
                        else:
                            st.write(output)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": output})
            
                elif flag == "Telemetry":
                    with st.chat_message("assistant"):
                        # st.write("Here is the chatbot response:")
                        if output.startswith('Please'):
                            st.write(output)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": output})
                        else:
                            st.write(output)
                            st.session_state.messages.append({"role": "assistant", "type": "text", "content": output})
            
                else:
                    with st.chat_message("assistant"):
                        # st.write("Here is the chatbot response:")
                        st.write(output)
                        st.session_state.messages.append({"role": "assistant", "type": "text", "content": output})
            

        # Display the "Clear Chat" button only if a query has been made
        if st.session_state.has_query:
            reset_button = st.button("Clear Chat")
            if reset_button:
                st.session_state.messages = []  # Clear the chat history
                st.session_state.has_query = False  # Reset query flag
                st.experimental_rerun()  # Rerun the app to reflect the cleared history
                    
    except Exception as e:
        err = f"An error occurred while calling the final function: {e}"
        return err

if __name__ == "__main__":
    main()


