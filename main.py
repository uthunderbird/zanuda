from crewai import Agent
from crewai import Task
from crewai import Crew
from crewai import Process
from langchain.agents import load_tools
from langchain_community.llms.openai import OpenAI

from scholar_search import read_papers
from scholar_search import search_in_google_scholar

investigator_tools = load_tools(["wikipedia"])

# Initialize SerpAPI tool with your API key
# os.environ["OPENAI_API_KEY"] = "Your Key"
# os.environ["SERP_API_KEY"] = "Your Key"


researcher = Agent(
  role='Research Analyst',
  goal='Analyze big amount of papers and answer question.',
  backstory='PhD in multiple disciplines. 60 years old respectable '
            'scientist.',
  tools=[read_papers],
  verbose=True,
  llm=OpenAI(temperature=0, model_name="gpt-4-1106-preview"),
)

investigator = Agent(
  role='Search engine investigator',
  goal='Find all required relevant information using available tools and '
       'save it into global context.',
  backstory="You're OSINT master and just curious about everything guy.",
  verbose=True,
  tools=investigator_tools + [search_in_google_scholar],
  allow_delegation=False,
  llm=OpenAI(temperature=0, model_name="gpt-4-1106-preview"),
)
writer = Agent(
  role='Writer',
  goal='Create engaging content',
  backstory="You're a famous multilingual pop-science writer, specialized on "
            "explaining hard questions in an easy, but comprehensive manner. "
            "Always ask on the same language on which the task was written.",
  verbose=True,
  allow_delegation=True,
  llm=OpenAI(temperature=0, model_name="gpt-4-1106-preview"),
)

# Create tasks for your agents
task0 = Task(description='Найти информацию о вреде жареного мяса',
             agent=investigator)
task00 = Task(description='Найти информацию о вреде алкоголя',
             agent=investigator)
task1 = Task(description='что более вредно - алкоголь или стейки?',
             agent=researcher)
task2 = Task(description='Напиши в пару абзацев результат анализа '
                         'сравнительного вреда '
                         'алкоголя и стейков. Используй ссылки на научные '
                         'работы.',
             agent=writer)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, investigator, writer],
  tasks=[task0, task00, task1, task2],
  verbose=True, # Crew verbose more will let you know what tasks are being worked on
  process=Process.sequential # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

if __name__ == '__main__':
    result = crew.kickoff()
