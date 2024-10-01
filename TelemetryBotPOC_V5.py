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
    You need to work as a telemetry expert and give important telemetry markers by understanding the provided user question and generating a relevant SQL query
 
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

CASE 1:    
- In the case 1, user will be asking for the telemetry for a particular action, here you need to provide all the columns in the select statement

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

CASE 2:
- In the case 2, user will be asking the telemetry for particular column/columns for a particular action, here you need to provide on the user asked columns in the select statement along with 'Platform', 'subWorkloadScenario' and 'Action_Type'

Example question and its SQL Query:
1)what is the value for module name for joining a meeting through calendar in ios
    SELECT [Platform],[subWorkLoadScenario],[Action_Type], [module_name] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Action_Type = 'Primary' AND Platform in ('IOS')
    2)Provide me module type, thread type and action gesture for adding a meeting to calendar in IOS and Android 
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_gesture],[module_type],[thread_type] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Platform in ('IOS', 'Android') AND Action_Type = 'Primary'
    3)For turning off the video in a meeting, what is the module summary, action outcome, panel type, action_scenario and action scenario type related to it
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_outcome],[panel_type],[action_scenario],[action_scenario_type],[module_summary] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Action_Type = 'Primary'
    4)action scenario, module type for join a meeting for maglev, web and ios
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_scenario],[module_type] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Action_Type = 'Primary' AND Platform ('Maglev','Web','IOS')
    5)for accepting a event, what is thready type, action outcome, action subworkload and module summary
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[thread_type],[action_outcome],[action_subWorkLoad],[module_summary] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Action_Type = 'Primary'
    6) give me panel type for meeting leaving in web and android
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[panel_type] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Action_Type = 'Primary' AND Platform in ('Web', 'Android')
    7) action scenario for community event joining
    SELECT [Platform],[subWorkLoadScenario],[Action_Type],[action_scenario] FROM df WHERE subWorkLoadScenario like "%user%asked%action%" AND Action_Type = 'Primary'

    If specifically mentioned android or maglev or ios or web, select the mentioned platform values in the where condition with platform column
    for ex: if asked for Web - Query: Platform = 'Web'
    for ex: if asked for Maglev and web - Query: Platform in ['maglev', 'Web']    
    

Take above examples as examples and generate query based on the users question

Very Very Important:
- They might be asking the questions in a very professional manner as well as in a uncommon speaking manner, understand what they are looking for by deeply analysing the users question and provide the response according to it
- If asked for a specific telemetry markers, don't provide all the telemetry markers, only provide the column which they have asked for
- If asked only for action scenario of a particular action/feature/subworkloadscenario, only provide the action scenario column values, don't give all the markers for that partciular action/feature/subworkloadscenario
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
            return "The telemetry you are requested for a particular action, doesn't have any data for it "
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
        7.Only give accurate result according to the question.
        
        Instructions for Generating SQL Query:
        1. Identify the user's request based on the keywords in the question.
        
        2. Handling "action" and "action scenario" in the question:
            -If both "action" and "action scenario" are mentioned in the question:
                -Consider one "action" on its own, which is not immediately followed by "scenario," as referring to the SubWorkLoadScenario column and use it for the output.
                Example:
                -User Question: "Can you provide the action where action scenario is storeNewMeeting is used?"
                SQL Query: SELECT DISTINCT platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'storenewmeeting';
                -User Question: "can you provide me action where action scenario is store New Meeting is used in any platform"
                SQL Query: SELECT DISTINCT platform,SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'storenewmeeting';
                
            
        3. Handling both "scenario" and "action scenario" in the question:
            -If the user mentions both "scenario" and "action scenario" in any part of the question, treat "scenario" alone (without being immediately preceeded by "action") as referring to the SubWorkLoadScenario column.
            -If "action" is followed by "scenario," treat it as referring to the action_scenario column.
             Example:
             - User Question: "Can you provide the scenario where action scenario is store NEw Meeting is used?"
             - SQL Query: `SELECT DISTINCT platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'storenewmeeting';
     
        4. Handling only "action" in the question:
            -If the user mentions only "action" in the question, treat "action" alone (without being immediately followed by "scenario") as referring to the SubWorkLoadScenario column.
             Example:
             - User Question: "can you provide me the action where automatic is used?"
             - SQL Query: `SELECT DISTINCT platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automatic';
     
        5. Handling only "scenario" in the question:
            -If the user mentions only "scenario" in the question, treat "scenario" alone (without being immediately preceeded by "action") as referring to the SubWorkLoadScenario column.
             Example:
             - User Question: "can you provide me the scenario where is automatic used?"
             - SQL Query: `SELECT DISTINCT platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automatic';
     
        6. Handling both "action" and "action gesture" in the {question}:
            -If the user mentions both "action" and "action scenario" in any part of the question, treat "action" alone (without being immediately followed by "scenario") as referring to the SubWorkLoadScenario column.
            -If "action" is followed by "gesture" treat it as referring to the action_gesture column.
             Example:
             - User Question: "can you provide me the action where action gesture is automatic"
             - SQL Query: `SELECT DISTINCT platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automatic';
     
        7. Handling both "scenario" and "action gesture" in the {question}:
            -If the user mentions both "scenario" and "action gesture" in any part of the question, treat "scenario" alone (without being immediately preceeded by "action") as referring to the SubWorkLoadScenario column.
            -If "action" is followed by "gesture," treat it as referring to the action_gesture column.
             Example:
             - User Question: "can you provide me the scenario where action gesture is automatic"
             - SQL Query: `SELECT DISTINCT platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automatic';

        8. Handling more than 1 word value as combined values:
            - If the value has more than 1 word, leave no space between words.
            Example: 
                -if user question contains "store new meeting" consider it as "storenewmeeting"
                -if user question contains "Update meeting" consider it as "updatemeeting"
                -if user question contains "modify stage View" consider it as "modifystageview"
    
        9. Handling uppercase characters:
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
            1. SELECT DISTINCT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'screenSharing';
            2. SELECT DISTINCT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'storenewmeeting';
            3. SELECT DISTINCT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'touch' AND module_name = 'savenewsession' AND module_type = 'pushbutton' AND action_outcome = 'send' AND platform = 'Android';
            
    
        Example:
        User Question: "Can you provide me the scenario where module name is Session concluded is used?"
        SQL Query should be like this: SELECT DISTINCT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE module_name = 'sessionconcluded';
    
        User Question: "Can you provide me the action where action scenario is entermeeting is used?"
        SQL Query should be like this: SELECT DISTINCT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_scenario = 'entermeeting';
        
        User Question: "can you provide me the scenario where action gesture is automated, module name is API workflow, module type is planner in android"
        SQL Query should be like this: SELECT DISTINCT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE action_gesture = 'automated' AND module_name = 'apiworkflow' AND module_type = 'planner' AND platform = 'Android';

        User Question: "module summary is Modifies meeting, what is the action related to it"
        SQL Query should be like this: SELECT DISTINCT refer_index, SubWorkLoadScenario,platform FROM df_copy WHERE module_summary = 'modifiesmeeting';
    
        User Question: "Can you provide the action where action scenario is storenewmeeting is used?"
        SQL Query should be like this: SELECT DISTINCT refer_index, SubWorkLoadScenario,platform FROM df_copy WHERE action_scenario = 'storenewmeeting';
    
        User Question: "action where thread type is planner"
        SQL Query should be like this: `SELECT DISTINCT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE thread_type = 'planner';

         User Question: "what action does primary schedule panel type related to"
        SQL Query should be like this: `SELECT DISTINCT refer_index, platform, SubWorkLoadScenario FROM df_copy WHERE panel_type = 'primaryschedule' 
    
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
            return "The action you are requested for the particular telemetry/markers, doesn't have any data for it"
        else:
            common_columns = inter_df.columns.intersection(sql_df.columns)
            matching_index = inter_df["refer_index"].tolist()
            output_df = sql_df.loc[matching_index,common_columns]
            result_df = output_df.drop(columns = ["refer_index"])
            result_df = result_df.reset_index(drop = True)
            result_df = result_df.drop_duplicates(keep='last')
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

    --Joining the meeting/Meeting - Joining  this is the category name you need to bucket where action or subworkloadscenario related to the list below.
  
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


    **Important: if the user gives the prompt of the existing feature analysis and understands it, bucket the action in the respective category and go inside the respective category, and inside that bucket, see related actions or subworkload scenarios.
    and according to that, first understand the pattern of those existing features and give the new telemetries for the new feature the user needed.**
     
        Examples:
       
            -Example query1: "Recommend telemetry for the '**New Feature Name**' feature which is under the Reaction Menu"
           
            Response template:
            platform:Maglev
           
            New Feature-**New Feature Name**
         
            **Based on the user query refer the relevent platform and provide the relvent of telemetry colums for that with the recommended values of you**
           
            platform:Ios
           
            New Feature-**New Feature Name**
         
           **Based on the user query refer the relevent platform and provide the  colums for that with the recommended values of you**
           
            platform:Andriod
           
            New Feature-**New Feature Name**
         
             **Based on the user query refer the relevent platform and provide the  colums for that with the recommended values of you**
         
            platform:Web
           
            New Feature-**New Feature Name**
         
            **Based on the user query refer the relevent platform and provide the colums for that with the recommended values of you**
         
            **If the user's query pertains to a specific platform, respond with information relevant only to that platform**
           
            New Feature-**New Feature Name**

            Example query2: "Recommend telemetry for the '**New Feature Name**' feature which is under the Reaction Menu for Maglev platform"
         
            Response template:
           
            PlatForm:which platform the user query is asking
            New Feature-**New Feature Name**
            **Refer to the repective all  column from action_gesture to subNav_entity_type   and give the recommended markers and its values**

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
        9. Give response only for the asked platforms if any of the platform is mentioned in the user query.
        10. If the user didn't mention any platform specifically or all is being mentioned in the user question, the response should be for all platforms.
        11. In any case of the question, the telemetries should be recommended.
	12. Provide only from the primary values where Action the column name is Action Type

        Strictly follow whatever mentioned above for any type of question.

     
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
        

        text = re.sub(r'[^a-zA-Z0-9\s":\n,]', '', text)
        result = ''
        inter_list = []
        for line in text.splitlines(keepends=True):
            if "json" not in line and "nan" not in line:
                
                inter_list.append(line)
                inter_list.append('\n')
                
        result = ''.join(inter_list)
   
        platform_line = re.search(r'(^Platform:\s.*)', result, re.MULTILINE)
        scenario_line = re.search(r'(^Scenario:\s.*)', result, re.MULTILINE)
        explanation_line = re.search(r'(^Explanation:)', result, re.MULTILINE)
        key_points_line = re.search(r'(^Key Points:)', result, re.MULTILINE)
        # Format the lines with Markdown for larger text and bold
        if platform_line:
            platform_line = platform_line.group(1)
            platform_line = f"## **{platform_line}**"  
        if scenario_line:
            scenario_line = scenario_line.group(1)
            scenario_line = f"## **{scenario_line}**"  
 
        if explanation_line:
            explanation_line = explanation_line.group(1)
            explanation_line = f"### **{explanation_line}**"  
 
        if key_points_line:
            key_points_line = key_points_line.group(1)
            key_points_line = f"### **{key_points_line}**"  
 
        final_result = result
        if platform_line:
            final_result = re.sub(r'(^Platform:\s.*)', platform_line, final_result, flags=re.MULTILINE)
        if scenario_line:
            final_result = re.sub(r'(^Scenario:\s.*)', scenario_line, final_result, flags=re.MULTILINE)
        if explanation_line:
            final_result = re.sub(r'(^Explanation:)', explanation_line, final_result, flags=re.MULTILINE)
        if key_points_line:
            final_result = re.sub(r'(^Key Points:)', key_points_line, final_result, flags=re.MULTILINE)
 
        return final_result
        
        
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
        Understand the user question and categorize the user question into one of the 4 categories -
            1 cat1: If the user prompts only for retrieval of telemetry (or) telemetry markers, telemetry for a certain column for any actions or scenarios, categorize it into "cat1" category
                -If user asks for specific column in a telemetry for a particular action, categorize that into "cat1" category
            Example questions - Give me the telemetry markers for join a meeting via calendar 
                                What is the telemetry for creating a community event in Maglev?
                                Can you provide markers for editing a meeting
                                give me the panel type for adding meeting to a calendar
                                what is the action gesture, module name and module type for sending applause reaction in web and ios
                                for canceling a meeting, what is the action scenario and action gesture
                                module name for community event joining
                                action scenario and thread type for leave a meeting
                                sharing screen's module type in maglev 
            IMPORTANT: 
            - Understand the question and identify, if user is seeking for telemetry markers or any other anamolies.
            - If the user wants telemetry/markers or any of the column values of any action from available subworkloadscenario, categorize into "cat1"
            - If the user is seeking for any anomolies other than telemetry markers, never categorize into "cat1"
            - If user seeks for action or any other anomolies other than telemetry/markers, do not categorize into "cat1"
            
                              
            2. cat2: If the user wants to retrieve action/subworkloadscenario by providing any of the telemetry/marker values by using its column names or column values, categorize it into "cat2"
            Example questions - Can you provide me the action where savenewmeeting is used?
                              what is the action related to the following marker, module type is meetings and panel type is meeting join?
                              Can you provide the action where action scenario is meetingjoin in Web platform?
                              give me scenario where session join category is used?
                              action scenario is entry meeting, what is the action realted to it?
                              action where module type is push button
                              action related to terminate meeting session action scenario
                              
            3. cat3 : If the user prompts are only related to suggest or recommend or create or generate, categorize them into the "cat3" category
            Example questions - Give me the recommended telemetry or marker for the new feature where it is inside the meeting feature?
                                Provide the telemetry or marker for the new feature embedded within that feature(meeting)?
                                I need the telemetry or marker  for the new feature that's included inside that feature?
                                Please share the telemetry data for the new feature inside?
                                Suggest the telemetry inputs for the new feature inside the meeting?
                                what markers are available for the new features?
                                Provide the specific telemetries for the new features?
                                What telemetry markers should we need for the newly implemented feature new sad reaction?
                                Provide the telemetry data for the new functionality jumping reaction
                                create telemetry for the feature qucik chat 
                                Please give me the telemetry markers related to the new feature within the joining a meeting
                                What telemetry or markers should we use for tracking the newly introduced feature hurry reaction
                                Please give me the telemetry markers related to the new feature for angry reaction
                                
            4. others:

            - You are an intelligent assistant responsible for answering telemetry-related queries. When responding:
            - If the user's question is general or unrelated to specific telemetry data (e.g., non-telemetry-related or open-ended), categorize the question as "Others" and respond accordingly.
            - For all other specific telemetry-related questions, provide accurate responses using the appropriate category (e.g., platform-specific telemetry).

                ***Any user prompt that does not directly relate to the three categories above—telemetry, subworkload scenarios, or telemetry recommendations—should be categorized as "others"
                
                Use the following criteria to categorize a question into "others":
                
                    -General or unrelated queries, as well as broad questions about telemetry, the dataset, or any queries that do not specifically request telemetry markers, actions, or scenarios.
                    -Non-retrieval and non-recommendation questions related to telemetries.
                    -Gibberish text or random, nonsensical input.
                    -Non-literal inputs, such as symbols or numbers without meaningful context.
                    -Empty inputs, consisting of only spaces or blank entries.
                    
                If the user question matches any of these criteria, it should be strictly categorized as "others" in any case of the question.
                Example questions:
                -What is telemetry
                -What is earth
                -What is subworkloadscenario

        Important notes:
        - All non retrieval questions and all non recommendation questions should be categorized under "others" in any case of the question.
        - The question which doesn't make any meaning or context also categorized under "others" in any case of the question.
        - The general question which is related to the dataset or telemetries should also be categorized under "others" in any case of the question.
        - Questions related to column names of the dataset should also be categorized as "others" in any case of the question.
        - Questions with no meaningfulness also be categorized as "others" in any case of the question.***
        
        **Output Format**:
            - The output should be either "cat1" or "cat2" or "cat3" or "others" based on the user question
        
    
        Example User Query and Expected Response:
        - **User Query**: "Give me the data for joining a meeting through chat button."
        - **Expected Response**: 
            - cat1
        - **User Query**: "what are the markers for presenting screen"
        - **Expected Response**: 
            - cat1
        - **User Query**: "what is the panel type and module name for editing a meeting"
        - **Expected Response**: 
            - cat1  
        - **User Query**: "Give me the markers for adding a meeting to calendar"
        - **Expected Response**: 
            - cat1
        - **User Query**: "For declining a meeting, what are the action gesture that will be popped up"
        - **Expected Response**: 
            - cat1
        - **User Query**: "For what scenario I should use ExpandableApp as Action subworkload"
        - **Expected Response**: 
            - cat2  
        - **User Query**: "target thread type is conference, what is the action related to it?"
        - **Expected Response**: 
            - cat2  
        - **User Query**: "action related to terminate meeting session action scenario"
        - **Expected Response**: 
            - cat2  
         **User Query**: "action where module type is push button"
        - **Expected Response**: 
            - cat2  
        - **User Query**: "For what action I should use messaging as its subnav entity type"
        - **Expected Response**: 
            - cat2
        - **User Query**: "I want the action associated with the following telemetry panel_type - stageSwitcher"
        - **Expected Response**: 
            - cat2
        - **User Query**: "Give me the recommended telemetry or marker for the new feature where it is inside the meeting feature."
        - **Expected Response**:
            -cat3
        - **User Query**: "I need the telemetry or marker for the new feature that's included inside that feature."
        - **Expected Response**:
            -cat3
        - **User Query**: "what markers are available for the new features?"
        - **Expected Response**:
            -others
        - **User Query**: "can you provide me count of unique action scenario for each platform"
        - **Expected Response**:
            -others
        - **User Query**: "Give me scenario where tfl is used"
        - **Expected Response**:
            -cat2            
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

        IMPORTANT: Always return the same appropriate category that you identified based on the user question in the response
            -Try to understand the question and categorize it accordingly
 
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
        #print(response)
        
        if "cat1" in response["output_text"].lower():
            return "Telemetry"
        elif "cat2" in response["output_text"].lower():
            return "Subworkloadscenario"
        elif "cat3" in response["output_text"].lower():
            return "Recommendation"
        elif "others" in response["output_text"].lower():
            return "others"          
    except Exception as e:
        return "Sorry, couldn't process your question. Try asking questions related to telemetry"


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
    st.set_page_config(layout="wide", menu_items = {'Get Help': 'mailto:v-shreyashaw@microsoft.com,v-ajithis@microsoft.com,v-rohithp@microsoft.com,v-saransu@microsoft.com?cc=v-ishashiny@microsoft.com','Report a Bug':'mailto:v-shreyashaw@microsoft.com,v-ajithis@microsoft.com,v-rohithp@microsoft.com,v-saransu@microsoft.com?cc=v-ishashiny@microsoft.com','About':"This Telemetry Chatbot has been fed with the telemetry data of meeting features of TFL and it can answer queries related to it. It can fetch telemetry markers for an existing action, fetch actions corresponding to telemetry markers, recommend telemetry for new features."} )

#codes to hide the git symbol on the top right cornor    
    hide_git = """
    <style>
    /* Hide the GitHub button */
    button[title="View on GitHub"] {
        display: none;
    }
    </style>
"""    

#codes to make the UI dynamic

    st.markdown("""
    <style>
        /* Chatbot container with dynamic width */
        .chatbot-container {
            width: 80%;  /* Flexible width that adjusts with screen size */
            max-width: 800px;  /* Max width to avoid stretching on large screens */
            min-width: 300px;  /* Minimum width for readability on small screens */
            margin: auto;  /* Center the container */
        }

        /* Dynamic font size for input and button */
        .chatbot-container input, .chatbot-container button {
            font-size: calc(16px + 0.2vw);  /* Dynamically scale font size */
        }
    </style>
    """, unsafe_allow_html=True)

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
    
#changing chatting alignment for dark theme
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

#changing chatting alignment for dark theme
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

#codes to make the question bar width dynamic for 2nd question
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

            elif (len(user_question)<15):
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
                print("hi")
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


