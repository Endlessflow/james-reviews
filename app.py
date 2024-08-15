from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
import requests
from dotenv import load_dotenv
import os
import openai
import json
import magic


# Load environment variables
load_dotenv()


def validate_env_vars():
    required_vars = [
        "OPENAI_API_KEY",
        "GITHUB_PERSONAL_ACCESS",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


# Validate environment variables
validate_env_vars()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_PERSONAL_ACCESS = os.getenv("GITHUB_PERSONAL_ACCESS")

# Initialize the OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define the state for the graph
class FileDiff(TypedDict):
    patch: str
    content: str


class PRInfo(TypedDict):
    description: str
    diffs: Dict[str, FileDiff]  # Keyed by filename


class PRState(TypedDict):
    pr_url: str
    pr_info: PRInfo
    context: str
    features: str
    team_review: str
    final_report: str


def gather_pr_info(state: PRState) -> PRState:
    # Extract repo info and PR number from the URL
    url_parts = state['pr_url'].rstrip('/').split('/')
    repo_owner = url_parts[-4]
    repo_name = url_parts[-3]
    pr_number = url_parts[-1]

    # Set up the API request
    api_url = f"https://api.github.com/repos/{
        repo_owner}/{repo_name}/pulls/{pr_number}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {GITHUB_PERSONAL_ACCESS}"
    }

    # Make the API request for basic PR data
    response = requests.get(api_url, headers=headers)
    pr_data = response.json()

    # Extract the PR description
    pr_description = pr_data.get("body", "")

    # Get the list of files changed
    files_url = pr_data.get("_links", {}).get(
        "self", {}).get("href") + "/files"
    files_response = requests.get(files_url, headers=headers)
    files_data = files_response.json()

    # Extract the diffs and file contents
    diffs = {}
    for file in files_data:
        filename = file['filename']
        patch = file.get('patch', '')
        raw_url = file['raw_url']

        # Download the file content
        file_content_response = requests.get(raw_url, headers=headers)
        file_content = file_content_response.content

        # Check MIME type to exclude binary files
        mime = magic.Magic(mime=True)
        content_type = mime.from_buffer(file_content)

        if 'text' not in content_type and 'json' not in content_type or filename.endswith('.excalidraw'):
            # Skip binary files
            continue

        diffs[filename] = {
            "patch": patch,
            "content": file_content.decode('utf-8')
        }

    # Update the state with the gathered PR info
    pr_info = {
        "description": pr_description,
        "diffs": diffs
    }

    return {"pr_info": pr_info}


def contextualize_info(state: PRState) -> PRState:
    # Prepare the prompt
    prompt = (
        f"Below is the information extracted from the PR:\n\n```json\n{json.dumps(state['pr_info'], indent=2)}\n```\n\nBased on the data provided, please analyze the following:\n1. What could be the possible motivation behind this PR?\n2. What specific problem or issue might this PR be trying to solve?\n3. How do the changes in the PR relate to the overall codebase? Consider potential impacts on functionality, performance, security, and maintainability.\n4. Speculate on the intent of the author. Why might they have made these specific changes? What are the potential benefits or risks?\n5. List any questions or concerns that arise from this PR, which might require further clarification from the author or additional review.\n\nPlease provide a detailed and thoughtful analysis, using all the available information and your expert knowledge."
    )

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert software engineer reviewing a pull request."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4095,
            temperature=0.6
        )

        # Extract the generated context from the response
        reply = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        reply = "Error generating context."

    print('context:', reply)
    return {"context": reply}


def extract_features(state: PRState) -> PRState:
    # Prepare the prompt
    prompt = f"Based on the following pull request information and contextual analysis, identify the new features or technical changes introduced by this PR: \n\nPR Information:\n```json\n{json.dumps(state['pr_info'], indent=2)}\n```\n\nContextual Analysis:\n```txt\n{state['context']}\n```\n\nPlease provide a detailed breakdown of the new features, technical changes, and any relevant implications."

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert software engineer reviewing a pull request."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4095,
            temperature=0.6
        )

        # Extract the generated context from the response
        reply = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        reply = "Error generating context."

    print('features:', reply)
    return {"features": reply}


def create_expert_checklist(state: PRState) -> PRState:
    # Prepare the prompt
    prompt = f"You are about to perform an expert-level review of a pull request. Below, you'll find detailed information about the PR, including the extracted context and identified features. You will adopt the mindset of three different expert personas, each focusing on a critical aspect of code quality.\n\nPR Information:\n```json\n{state['pr_info']}\n```\n\nContextual Analysis:\n```\n{state['context']}\n```\n\nIdentified Features:\n```\n{state['features']}\n```\n\nPlease provide a detailed review from each of the following perspectives:\n\n1. **Code Quality Guru**\n   - Focus on readability, maintainability, and overall structure. Consider best practices, documentation, and ease of understanding. Identify issues or areas for improvement in code quality and maintainability.\n\n2. **Performance Wizard**\n   - Focus on performance and efficiency. Consider speed, resource usage, scalability, and identify bottlenecks or inefficiencies. Suggest optimizations if necessary.\n\n3. **Security Sentinel**\n   - Focus on security and compliance. Consider security vulnerabilities, data handling, and industry standards compliance. Highlight security risks or concerns and suggest improvements.\n\n**Review Structure**:\nEach persona should provide a detailed, thoughtful review with specific points that a human reviewer might want to double-check or consider further. Each review should include the persona name and a thorough analysis of the PR from their perspective.\n\nCode Quality Guru Review:\nName: Code Quality Guru\nReview:\n[Your analysis here]\n\nPerformance Wizard Review:\nName: Performance Wizard\nReview:\n[Your analysis here]\n\nSecurity Sentinel Review:\nName: Security Sentinel\nReview:\n[Your analysis here]\n\nThink deeply about the potential motivations behind the PR, the implications of the changes, and any areas that could be optimized or improved."

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI system well versed in software engineer. You are currently reviewing a pull request."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4095,
            temperature=0.6
        )

        # Extract the generated context from the response
        reply = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        reply = "Error generating context."

    print('team_review:', reply)
    return {"team_review": reply}

def final_review(state: PRState) -> PRState:
    # Prepare the prompt
    prompt = f"""You are tasked with performing an expert-level review of a pull request (PR) using information compiled. Below is structured data about the PR, including context, key features, and evaluations from a team of different AI agents.\n\n**PR Data:**\n```json\n{state}\n```\n\n**Task:**\nYour goal is to create a focused and highly actionable PR review. The review should prioritize specific, targeted feedback that directly addresses key pain points in the code, offering immediate suggestions for improvement. Each comment should be quoting a specific line or section of the code, aiming for clarity and impact.\n\n**Guidelines for the Review:**\n\n1. **Contextual Summary:**\n   - Briefly summarize the purpose of the PR, including the assumptions made on its context, the problem it seems to address, and the key changes introduced.\n\n2. **Key Changes Overview:**\n   - List the primary changes made in the PR, providing a clear, high-level but thorough understanding of the modifications.\n\n3. **Focused, Actionable Comments:**\n   - Provide a series of specific, actionable comments that address identified issues or areas for improvement in the code.\n   - Each comment should:\n     - Reference a specific line or section of the code.\n     - Clearly explain the issue or potential improvement.\n     - Offer a concrete suggestion or solution that can be implemented immediately.\n\n4. **General Cautions (Optional):**\n   - If applicable, include a section for general cautions or remarks that don’t fit into the specific comments but are important for future consideration. These should still be concise and actionable where possible.\n\n5. **Approval Recommendation:**\n   - Conclude with a clear recommendation:\n     - [ ] Request Changes: Significant issues need to be addressed.\n     - [ ] Approve with Comments: Minor changes or clarifications are recommended.\n     - [ ] Approve: The PR is ready for merge.\n\n**Formatting Template:**\n\n```\n**PR Review Summary**\n\n**Title:** [Concise PR title]\n\n**Context:**\n- [Brief summary of the PR’s purpose and context]\n\n**Key Changes:**\n- [List of key changes made in the PR]\n\n**Specific Comments:**\n[Detailed, actionable comment on a specific issue]\n```\ncode X\n```\n\n[Detailed, actionable comment on a specific issue]\n```\ncode Y\n```\n\n...\n\n**General Cautions:**\n- [Optional section for general remarks or cautions]\n\n**Approval Recommendation:**\n- [Choose one: Request Changes / Approve with Comments / Approve]\n```"""

    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI entity with the goal of reducing suffering in the universe, increasing prosperity in the universe and increasing understanding in the universe."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4095,
            temperature=0.7
        )

        # Extract the generated context from the response
        reply = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        reply = "Error generating context."

    print('final report:', reply)
    return {"final report": reply}    


# Create the state graph
workflow = StateGraph(PRState)

# Add nodes to the graph
workflow.add_node("gather_pr_info", gather_pr_info)
workflow.add_node("contextualize_info", contextualize_info)
workflow.add_node("extract_features", extract_features)
workflow.add_node("create_expert_checklist", create_expert_checklist)
workflow.add_node("final_review", final_review)

# Define the edges
workflow.set_entry_point("gather_pr_info")
workflow.add_edge("gather_pr_info", "contextualize_info")
workflow.add_edge("contextualize_info", "extract_features")
workflow.add_edge("extract_features", "create_expert_checklist")
workflow.add_edge("create_expert_checklist", "final_review")
workflow.add_edge("final_review", END)

# Compile the graph
app = workflow.compile()

# Test the graph with an initial state
initial_state = {
    "pr_url": "https://github.com/ai-cfia/fertiscan-backend/pull/106",
    "pr_info": "",
    "context": "",
    "features": "",
    "team_review": "",
    "final_report": ""
}

result = app.invoke(initial_state)


def write_file(content: str, filename: str):
    with open(filename, 'w') as file:
        file.write(content)


# dump results
write_file(json.dumps(result), 'result.json')

# write report
write_file(result['final_report'], 'final_report.md')

