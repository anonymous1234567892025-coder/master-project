# The documentation of Azure, Neo4j ,arXiv, Semantic Scholar API assisted me with the creation of code
# The tutorials from W3 Schools aided me with String Manipulation and Comparison (https://www.w3schools.com/)
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
from neo4j import GraphDatabase  
import arxiv
import requests
import re
from marker.converters.pdf import PdfConverter
from io import BytesIO
from marker.models import create_model_dict
from time import sleep
import sys
import unicodedata
from copy import deepcopy
import json
from decimal import Decimal


#Semantic API Connection
SEMANTIC_API_ENDPOINT = "OMITTED"
SEMANTIC_API_KEY = "OMITTED"
SEMANTIC_API_HEADERS = {"x-api-key": SEMANTIC_API_KEY} 

# Azure Storage Connection
AZURE_STORAGE_CONNECTION = "OMITTED"
PDF_PAPER_VERSION = "OMITTED"
MARKDOWN_PAPER_VERSION = "OMITTED"

blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION)
pdf_paper_version_container_client = blob_service_client.get_container_client(PDF_PAPER_VERSION)
md_paper_version_container_client = blob_service_client.get_container_client(MARKDOWN_PAPER_VERSION)

# Neo4j Connection
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "OMITTED"
NEO4J_PASSWORD = "OMITTED"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER,NEO4J_PASSWORD))

#Azure OpenAI connection
openai_endpoint="OMITTED"
openai_deployment ="OMITTED"
openai_subscription_key ="OMITTED"
openai_api_version = "OMITTED"

#Azure OpenAI connection Client
openai_client = AzureOpenAI(
    api_key= openai_subscription_key,
    api_version=openai_api_version,
    azure_endpoint=openai_endpoint
)

arxiv_ids = []

# This functions downloads the papers from arxiv API
# The Arxiv ID is retreived using Regex and stored for later use
# Then stores it Azure Storage Container
def get_arxiv_papers(arxiv_ids,pdf_paper_version_container_client):
    number_of_papers_collected = 0
    search = arxiv.Search(
    query = '"self driving cars" OR "autonomous vehicles"', 
    sort_by= arxiv.SortCriterion.Relevance,
    sort_order = arxiv.SortOrder.Descending
    )


    for result in search.results():
        print(f"Paper Name: {result.title}")
        extract_arxiv_id_format = r"^(\d{4}\.\d{4,5})(?:v\d+)?$"
        check_arxiv_id = re.match(extract_arxiv_id_format, str(result.get_short_id()))
        if check_arxiv_id:
            print (f"Correct Arxiv ID {check_arxiv_id.group(1)}")
            arxiv_ids.append(check_arxiv_id.group(1))
        blob_client = pdf_paper_version_container_client.get_blob_client(blob = f"{check_arxiv_id.group(1)}.pdf")

        paper_data = requests.get(result.pdf_url, stream=True)
        blob_client.upload_blob(data=paper_data.raw, overwrite=True) 

        number_of_papers_collected = number_of_papers_collected +1
        if number_of_papers_collected ==50:
            # return "stop"
            break

    print(f"The number of papers downloaded to pdfcontainer is  {number_of_papers_collected}")

    return arxiv_ids
    
# This function keeps the abstract,introduction,conclusion,summary and references
# By going through the markdown file line by line
# Adds the selected to sections to reduced_markdown

def keep_selected_sections(md):
    reduced_markdown = ""
    paper_lines = md.markdown.split('\n')

    present_area = ""
    present_text = []
    selected_parts = ("abstract", "introduction", "conclusion","summary")
    reference_title = ("references", "bibliography" , "works cited")

    # This function gets the original lines from the markdown which has the reference section
    # Then uses regex to detect when the new reference starts
    # Returns the cleaned reference strings
    def get_references_markdown(references):
        elements = []
        current = None
        check_start_reference = re.compile(r'^\s*(?:\[(\d+)\]|(\d+)[\.\)]?)\s+(.*\S)?\s*$')
        for line in references:
            clean_line = line.rstrip()
            if not clean_line.strip():
                continue
            check_line_reference = check_start_reference.match(clean_line)
            if check_line_reference:
                if current:
                    elements.append(current.strip())
                current = (check_line_reference.group(3) or "").strip()
            else:
                if current is None:
                    current = clean_line.strip()
                else:
                    current += " "+ clean_line.strip()
        if current:
            elements.append(current.strip())
        return elements
    
    # This function takes the section title and the markdown lines within that section
    # Then makes decision if the text in the reference section or if it belongs to the other specfied sections
    def clean_markdown_section(section_title,markdown_lines):
        nonlocal reduced_markdown
        text = "\n".join(markdown_lines).strip()
        if not text:
            return
        small_section_title = section_title.lower()
        if any(part in small_section_title for part in selected_parts):
            reduced_markdown += f"## {section_title}\n\n{text}\n\n"
        elif any(part in small_section_title for part in reference_title ):
            cleaned_references = get_references_markdown(markdown_lines)
            if cleaned_references:
                reduced_markdown += f"## {section_title}\n\n" + "\n" .join(f"{a+1}. {b}" for a,b in enumerate(cleaned_references)) + "\n\n"
        
    for paper_line in paper_lines:
        if paper_line.startswith('#'):
            if present_area:
                clean_markdown_section(present_area,present_text)
            present_area = paper_line.lstrip('#').strip()
            present_text = []
        else:
            present_text.append(paper_line)

    if present_area:
        clean_markdown_section(present_area,present_text)
        

    return reduced_markdown.strip()

# Gets the PDFs from the PDF container in Azure Storage 
# Converts the PDFs into Markdown using the marker-pdf library
# Stores the Markdowns to MD container in Azure Storage
def convert_pdf_to_md():
    print(" Starting to convert paper pdf versions to markdown")
    model_dictionary = create_model_dict()
    pdf_converter = PdfConverter(model_dictionary)

    for blob in pdf_paper_version_container_client.list_blobs(name_starts_with=""):
        pdf_paper_name = blob.name
        md_paper_name = pdf_paper_name.replace(".pdf",".md")
        if any (md_blob.name == pdf_paper_name for md_blob in md_paper_version_container_client.list_blobs(name_starts_with="")):
            print(f" Paper Conversion Done Already: {pdf_paper_name}")
            continue
        
        print(f"Processing Conversion {pdf_paper_name}")

        pdf_client_azure = pdf_paper_version_container_client.get_blob_client(pdf_paper_name)
        pdf_data = pdf_client_azure.download_blob().readall()

        markdown_data = pdf_converter(BytesIO(pdf_data))
        markdown_text = keep_selected_sections(markdown_data)


        markdown_client_azure = md_paper_version_container_client.get_blob_client(md_paper_name)
        markdown_client_azure.upload_blob(markdown_text, overwrite=True)

        print(f"{md_paper_name} Markdown Saved")


# Adds the paper node with metadata with arXIv ID to Neo4j database by using the output of Semantic API. 
# The cypher query is ran to add the node to the graph
def add_paper(response_data,arxiv_id):
    print(response_data.get('title'))
    print(arxiv_id)

    author_names = [author["name"] for author in response_data.get("authors")]
    summary = driver.execute_query("""
        CREATE (a:Paper {PaperId: $name, ArxivId: $arxivid,Title: $title, PublicationDate: $publicationDate, Authors :$authors, Url: $url})
        """,
        name=response_data.get("paperId"), arxivid=arxiv_id, title=response_data.get("title"), publicationDate=response_data.get("publicationDate"), authors= author_names, url = response_data.get("url"),
        database_="paperconnectionthree",
    ).summary
    print("Created {nodes_created} nodes in {time} ms.".format(
        nodes_created=summary.counters.nodes_created,
        time=summary.result_available_after
    ))

    return arxiv_id

# Queries the semantic API with the arXiv ID passed and query parameterss
# Returns the metadata for a given arXiv ID.    
def query_semantic_api(arxiv_id,SEMANTIC_API_HEADERS,SEMANTIC_API_ENDPOINT):
    print(type(arxiv_id))

    url = f"{SEMANTIC_API_ENDPOINT}/paper/ARXIV:{arxiv_id}"

    query_params = {"fields": "title,url,publicationDate,authors,url"}
    
    response = requests.get(url, params=query_params, headers=SEMANTIC_API_HEADERS)

    if response.status_code == 200:
        response_data = response.json()
        return response_data, arxiv_id
    else:
        print(f"Request failed with status code {response.status_code} : {response.text}")
        return "Semantic Api Failed"

# Queries the semantic API at a different endpoint to get the references for a given arXiv ID
# Returns the references
def get_references_from_paper(arpaper_id,SEMANTIC_API_HEADERS,SEMANTIC_API_ENDPOINT):
    sleep(3)

    paperid = arpaper_id
    url = f"{SEMANTIC_API_ENDPOINT}/paper/ARXIV:{paperid}/references?limit=1000&offset={0}"
    query_params = {"fields": "title,authors,publicationDate,url"}
    response = requests.get(url, params=query_params, headers=SEMANTIC_API_HEADERS)
    if response.status_code == 200:
        response_data = response.json()
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

    paper_references = []
    for i,ref in enumerate(response_data["data"],start=1):
        cited_paper = ref["citedPaper"]
        paper_references.append({
            "index" : i,
            "title" : cited_paper.get("title","N/A"),
            "authors": [a["name"] for a in cited_paper.get("authors",[])],
            "paperId" : cited_paper.get("paperId","N/A"),
            "publicationDate" : cited_paper.get("publicationDate", "N/A"),
            "url" : cited_paper.get("url","N/A")
        
        })
    print(f"{paper_references}")

    return paper_references, arpaper_id
    
# This function normalises a given string so strings are in small letters with no punctuations or accents.
def normalise_string(string):
    string = ''.join(a for a in unicodedata.normalize('NFKD', string or '') if not unicodedata.combining(a)).lower()
    string = re.sub(r"[^\w\s]", " ", string)
    return re.sub(r"\s+", " ",string).strip()

# This functions removes the html span tags and returns it as a plain text.
def remove_span(string):
    string = re.sub(r"-\s*<span[^>]*></span>\s*$","",string)
    string = re.sub(r"<span[^>]*>","",string).replace("</span>","")
    string = re.sub(r"\*([^*]+)\*", r"\1",string)
    string = string.replace("`","")
    return string.strip()

# This function structures the references from the markdown with index,title and authors
def structure_references(references):
    quote= r'["“”«»]'
    structured_references = []

    for each_reference in references:
        removed_span = remove_span(each_reference)

        b = re.match(r"\[(\d+)\]\s*(.*)", removed_span)
        if not b:
            continue
        reference_id = int(b.group(1))
        reference_body = b.group(2).strip()

        reference_title_search = re.search(fr"{quote}(.*?){quote}",reference_body)
        if reference_title_search:
            title = reference_title_search.group(1).strip()
            authors = reference_body[:reference_title_search.start()].strip(",;-")
        else:
            separate = len(reference_body)
            inside_body = re.search(r"\bin\b", reference_body)
            if inside_body: separate = min(separate, inside_body.start())
            dot_search = re.search(r"\.", reference_body)
            if dot_search: separate = min(separate,dot_search.start())
            title = reference_body[:separate].strip(",;-")
            authors=""

        authors = authors.replace("——", "").strip(",;-")

        final_authors = [d.strip().replace("’",",") for d in authors.split(",") if d.strip()]

        if title:
            structured_references.append({"index": reference_id, "title": title,"authors" : final_authors})

        structured_references = [f for f in structured_references if f["title"]]
        structured_references.sort(key= lambda d: d ["index"])

        return structured_references

# The titles from the markdown references and semantic scholar api is matched
# The missing data from semantic scholar API is retrieved from the markdown references
# Returns a sorted list 
def merging_references(structured_references,semantic_output_references):
    structured_references = structured_references or []
    semantic_output_references = semantic_output_references or []
    title_check = {normalise_string(a["title"]): a for a in structured_references if a.get("title")}
    done = set()
    joined = []
    for semantic_reference in semantic_output_references:
        if not semantic_reference.get("authors") and not semantic_reference.get("paperId"):
            continue
        location_key = normalise_string(semantic_reference.get("title",""))
        if location_key in title_check:
            cleaned_referenced = title_check[location_key]; done.add(location_key)
            semantic_reference = {**semantic_reference, "title" : cleaned_referenced["title"], "authors" : cleaned_referenced["authors"], "index" : cleaned_referenced["index"] }
        joined.append(semantic_reference)

    for j, cleaned_referenced in title_check.items():
        if j not in done:
            joined.append({
                "index" : cleaned_referenced["index"],
                "title" : cleaned_referenced["title"],
                "authors" : cleaned_referenced["authors"],
                "paperId" : None
            })
    return sorted(joined, key=lambda b: b.get("index", 10**9))

# Gets the markdown references and structures the markdown references for cross checking
# Merges the data from the semantic out API and the markdown references
def get_both_references(semantic_output_references,arxiv_id):
    
    blob_client = md_paper_version_container_client.get_blob_client(f"{arxiv_id}.md")
    md_data = blob_client.download_blob().readall()
    md_text = md_data.decode("utf-8")
    markdown_references = md_text.split("## REFERENCES", 1)[-1]
    markdown_references_find = re.findall(r"\[\d+\].*?(?=(?:\[\d+\]|$))",markdown_references, re.DOTALL)
    markdown_references_find = [each_reference.strip().replace("\n", " ") for each_reference in markdown_references_find ]
    
    cleaned_references = structure_references(markdown_references_find)
    joined_references = merging_references(cleaned_references,semantic_output_references)

    return joined_references,arxiv_id

# Sending the merged references to the LLM for Citation Filtration
# The LLM responses for a given paper is logged in a text file
def llm_filter_references(references,arxiv_id):
    references_before = sum(1 for reference in references if "index" in reference)    
    prompt= (f"""
        You are an expert on autonomous vehicles
        The references of a paper will be passed in. Filter the citations so only the citations that is relevant to autonomous vehicles is kept
        Return Guidance:
        Return the references in a valid JSON Object with key called filtered_references
        {references}   
    """          
    )
    llm_response = openai_client.chat.completions.create(
        model = openai_deployment,
        temperature = 0.0,
        response_format = {"type" : "json_object"},
        messages = [{"role" : "user" , "content" : prompt}]
    )

    llm_message =  llm_response.choices[0].message.content.strip()

    print(llm_message)


    llm_selected_references = json.loads(llm_message)

    references_after = sum(1 for reference in llm_selected_references["filtered_references"] if "index" in reference)    



    with open(f"references_count_with_gpt_4_1.txt","a",encoding="utf-8") as b:
        b.write(f"Paper Id : {arxiv_id}\n")
        b.write(f"References_Filter_Before : {references_before}\n")
        b.write(f"References_Filter_After : {references_after}\n")
        b.write("-------------------------------------------------------------------------------------\n")


    return llm_selected_references,arxiv_id

# Adds the references selected from the LLM responses to the Neo4j Database
# The references are added to the citing paper by matching by the arxiv id 
# So Neo4j knows which nodes to add the references to
def add_references_database(llm_selected_references, arxiv_id):
    summary = driver.execute_query(
        """
        MATCH (d:Paper {ArxivId: $arxivId})
        RETURN d.PaperId AS paperId
        """,
        arxivId = arxiv_id,
        database_="paperconnectionthree",
    )
    citing_paper_id = summary.records[0]["paperId"]
    for reference in llm_selected_references.get("filtered_references",[]):

        cited_paper_id = reference.get("paperId")
        if not cited_paper_id:
            continue
        title = reference.get("title")
        authors = reference.get("authors")
        publication_date = reference.get("publicationDate")
        Url = reference.get("url")



        driver.execute_query(
            """
            MATCH(b: Paper {PaperId: $sourceId})

            MERGE (c: Paper  {PaperId: $targetId})
            ON CREATE SET c.CreatedAt = timestamp()
            SET c.Title = coalesce($title, c.Title)
            SET c.Authors = coalesce($authors, c.Authors)
            SET c.PublicationDate = coalesce($publicationDate, c.PublicationDate)
            SET c.Url = coalesce($url, c.Url)

            MERGE (b) - [:REFERENCES] -> (c)
            """,

            targetId = cited_paper_id,
            title = title,
            authors=authors,
            publicationDate = publication_date,
            sourceId = citing_paper_id,
            url =Url,
            database_="paperconnectionthree",



        ).summary
    
    return arxiv_id

# Preparing for the Second Prompt to the LLM for hypothesis and research question extracion
# The references are removed from the markdown text to reduce tokens
def delete_references(markdown_text):
    format = r'(?mi)^\s{0,3}##\s*(?:<[^>]*>\s*)*(?:\*\*)?\s*references\s*(?:\*\*)?\s*$'
    find_references = re.search(format,markdown_text)
    return markdown_text[:find_references.start()] if find_references else markdown_text

# Preparing for the Second Prompt to the LLM for hypothesis and research question extracion
# The references are removed from the markdown text to reduce tokens
# The LLM responses for a given paper is logged in a text file
def ask_llm_hypotheis_rq(arxiv_id,md_paper_version_container_client):
    markdown_paper = md_paper_version_container_client.download_blob(f"{arxiv_id}.md")
    removed_references = markdown_paper.content_as_text()
    removed_references = delete_references(removed_references)

    prompt = (f"""
        You are an expert on automonomous vehicles
        An markdown file is passed with abstract/introduction/conclusion/summary
        Extract the hypothesis, if there is no hypothesis,then extract the research question from markdown file
        Extract the results from the file
        Decide if the hypothesis has been met or research question has been met
        Return ("Reasoning") Why has the hypothesis/research question has been  met or not ?
        Return in Valid JSON
        Extracted Hypothesis (What is Extracted Hypothesis Text?)
        Extracted Research Question (What is Research Question Text?)
        Hypothesis_Met (True/False)
        Research_Question_Met (True/False)
        {removed_references}     
               
              
        """  
        )
    
    llm_response = openai_client.chat.completions.create(
        model = openai_deployment,
        temperature = 0.0,
        response_format = {"type" : "json_object"},
        messages = [{"role" : "user" , "content" : prompt}]
    )
    
    llm_message =  llm_response.choices[0].message.content.strip()
    llm_returned_info = json.loads(llm_message)

    with open(f"hypothesis_research_question_count_with_gpt_4_1.txt","a", encoding="utf-8") as b:
        b.write(f"Paper ID: {arxiv_id}\n")
        b.write(f"extracted_hypothesis: {llm_returned_info['Extracted_Hypothesis']}\n")
        b.write(f"extracted_research_question: {llm_returned_info['Extracted_Research_Question']}\n")
        b.write(f"hypothesis_met: {llm_returned_info['Hypothesis_Met']}\n")
        b.write(f"research_question_met: {llm_returned_info['Research_Question_Met']}\n")
        b.write(f"reasoning: {llm_returned_info['Reasoning']}\n")
        b.write("-------------------------------------------------------------------------------------\n")

    print(llm_returned_info)


    return llm_returned_info,arxiv_id
# This function uploads the hypothesis and research question for a given paper
def upload_hypotheis_and_reasoning(llm_returned_info,arxiv_id):
    hypothesis = (llm_returned_info.get("Extracted_Hypothesis") or "").strip()
    check_return = (llm_returned_info.get("Extracted_Research_Question"), "")
    if isinstance(check_return,str):
        research_q= check_return.strip()
    elif isinstance(check_return,(list,tuple)):
        research_q = " ".join(str(b).strip() for b in check_return if b ).strip()
    else:
        research_q= str(check_return).strip()



    met_hypothesis = llm_returned_info.get("Hypothesis_Met")
    met_research = llm_returned_info.get("Research_Question_Met")

    text_reasoning = llm_returned_info.get("Reasoning") or {}
    if isinstance(text_reasoning,dict):
        reason_hypothesis = text_reasoning.get("Hypothesis_Met","") or ""
        reason_research =  text_reasoning.get("Research_Question_Met", "") or ""
    else:
        reason_hypothesis = str(text_reasoning)
        reason_research = str(text_reasoning)

    driver.execute_query(
    """
    MATCH (a:Paper {ArxivId: $arxivId})
    WITH a,
        $hypothesis as hypothesis,
        $research_q as research_q

        



    FOREACH (_ IN CASE WHEN hypothesis <> '' THEN [1] ELSE [] END |
        MERGE (hypo:Hypothesis {paperArxivId: $arxivId, text: hypothesis})
        MERGE (a)-[:HAS_HYPOTHESIS]->(hypo)
        SET hypo.Met = $met_hypothesis,
            hypo.Reasoning = $reason_hypothesis
    )

    FOREACH(_ IN CASE WHEN research_q <> '' THEN [1] ELSE [] END |
        MERGE (research:ResearchQuestion {paperArxivId: $arxivId, text: research_q} )
        MERGE (a)-[:HAS_RESEARCH_QUESTION]->(research)
        SET research.Met = $met_research,
            research.Reasoning = $reason_research
            

    )    




    """,
    arxivId=arxiv_id,
    hypothesis =hypothesis,
    research_q = research_q,
    met_hypothesis = met_hypothesis,
    reason_hypothesis=reason_hypothesis,
    met_research=met_research,
    reason_research=reason_research,
    database_='paperconnectionthree',
    )
    
    return




if __name__ == "__main__":

    # This is where all the functions are called
    # The papers are downloaded by arXiv API is logged in a text file with their IDs
    # Every 10 papers processed the code will ask to carry on or not
    # Because of the token limitation on the models

    arxiv_ids= get_arxiv_papers(arxiv_ids,pdf_paper_version_container_client)
    convert_pdf_to_md()
    for arxiv_id in arxiv_ids:
            with open(f"paper_list.txt","a", encoding="utf-8") as b:
                b.write(f"{arxiv_id}\n")


    arxiv_ids = []
    with open(f"paper_list.txt","r", encoding="utf-8") as b:
        for row in b:
            arxiv_ids.append((row).strip())

    number=0

    for arxiv_id in arxiv_ids:
        if number % 10 ==0:
            user_information = input(f" {number} papers processed. Carry on or not? (Check Azure usuge) (y/n)")
            if user_information not in ("y"):
                print("stopping")
                sys.exit(0)
        number = number +1
        response_data, arxiv_id = query_semantic_api(arxiv_id,SEMANTIC_API_HEADERS,SEMANTIC_API_ENDPOINT)
        arxiv_id = add_paper(response_data,arxiv_id)
        paper_references,arxiv_id=get_references_from_paper(arxiv_id,SEMANTIC_API_HEADERS,SEMANTIC_API_ENDPOINT)
        joined_references, arxiv_id = get_both_references(paper_references,arxiv_id)
        final_references,arxiv_id = llm_filter_references(joined_references,arxiv_id)
        arxiv_id=add_references_database(final_references,arxiv_id)
        information, arxiv_id = ask_llm_hypotheis_rq(arxiv_id,md_paper_version_container_client)
        upload_hypotheis_and_reasoning(information,arxiv_id)
        








    

    

