from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Process, Crew
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()


# Loading Human Tools
# human_tools = load_tools(["human"])

"""
- define agents that are going to research latest AI tools and write a blog about it 
- explorer will use access to internet to get all the latest news
- writer will write drafts 
- critique will provide feedback and make sure that the blog text is engaging and easy to understand
"""
explorer = Agent(
    role="Senior Technology Journalist Researcher",
    goal="""Find and explore the most exciting news, projects of {company} 
    in the ai and machine learning space in the last week of may 2024. 
    You MUST find the newest and the latest news. 
    You must use the search tool to find the most exciting news and the scrape tool to scrape the data.""",
    backstory="""You are and Expert strategist that knows how to spot emerging trends and companies in AI, tech and machine learning. 
    You're great at finding interesting, exciting projects on most important technology news portals. You turned scraped data into detailed reports with names
    of most exciting projects an companies in the ai/ml world. ONLY use scraped data from the internet for the report.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool, scrape_tool],
)

writer = Agent(
    role="Creative Content Creator",
    goal="Write engaging and interesting blog post about latest news on AI and machine learning from the report.",
    backstory="""You are an Expert Writer on technical innovation, especially in the field of AI and machine learning. You know how to write in 
    engaging, interesting but simple, straightforward and concise. You know how to present complicated technical terms to general audience in a 
    fun way by using layman words.ONLY use scraped data from the internet for the blog.""",
    verbose=True,
    tools=[scrape_tool],
    allow_delegation=True,
)
critic = Agent(
    role="Expert Writing Critic",
    goal="Provide feedback, criticize and suggest the new text to the blog post draft. Make sure that the tone and writing style is compelling, simple and concise. Make sure the links in the blog post are the same as scraped before.",
    backstory="""You are an Expert at providing feedback to the content creators. You can tell when a blog text isn't concise,
    simple or engaging enough. You know how to provide helpful feedback that can improve any text. You know how to make sure that text 
    stays technical and insightful by using layman terms.
    """,
    verbose=True,
    # allow_delegation=True,
)

task_report = Task(
    description="""Use and summarize scraped data from the internet to make a detailed report 
    on the latest rising projects in AI for {company}. Use ONLY scraped data to generate the report. 
    Your final answer MUST be a full analysis report, text and the url reference only, 
    ignore any code or anything that isn't text or the url. 
    """,
    expected_output="""The report has to have bullet points and with 5-10 exciting news, 
    AI projects, and tools. Write names of every tool and project. 
    Each bullet point MUST contain 3 sentences that refer to one specific news, 
    ai company, product, model or anything you found on the internet.""",
    agent=explorer,
    output_file="report.md",
)

task_blog = Task(
    description="""Write a blog article about the latest AI tools for {company}
    with text only and with a short but impactful headline and at least 10 paragraphs. 
    Blog should summarize the report on latest ai tools found on the internet. 
    The title of the blog post should be engaging and compeling. 
    Style and tone should be compelling and concise, fun, technical but also use layman words for the general public. 
    The conclusion must summarize the most important points, and make sure call the user to like and comment on the blog post.
    Don't write "**Paragraph [number of the paragraph]:**", instead start the new paragraph in a new line. 
    Write names of projects and tools in BOLD.
    ALWAYS include links to projects/tools/research papers and urls. ONLY include information from important technology portals.
    """,
    expected_output="""For your Outputs use the following markdown format:
    ```
    ## [Title of the blog post]
    - [Introduction]
    ## [Title of post](link to project)
    - [Interesting facts]
    - [Own thoughts on how it connects to the overall theme of the newsletter]
    ## [Title of second post](link to project)
    - [Interesting facts]
    - [Own thoughts on how it connects to the overall theme of the newsletter]
    [Similar for the rest of the posts...]
    ## Conclusion
    - [Summarize the most important points, and make sure call the user to like and comment on the blog post.]
    ```
    """,
    agent=writer,
    output_file="blog.md",
)

task_critique = Task(
    description="You must critique the blog post draft. Make sure that the tone and writing style is compelling, simple and concise.",
    expected_output="""The Output MUST have the following markdown format:
    ```
    ## [Title of the blog post]
    - [Introduction]
    ## [Title of post](link to project)
    - [Interesting facts]
    - [Own thoughts on how it connects to the overall theme of the newsletter]
    ## [Title of second post](link to project)
    - [Interesting facts]
    - [Own thoughts on how it connects to the overall theme of the newsletter]
    [Similar for the rest of the posts...]
    ## Conclusion
    - [Summarize the most important points, and make sure call the user to like and comment on the blog post.]

    ```
    Make sure that it does and if it doesn't, rewrite it accordingly.
    """,
    agent=critic,
    output_file="critique.md",
)

# instantiate crew of agents
crew = Crew(
    agents=[explorer, writer, critic],
    tasks=[task_report, task_blog, task_critique],
    verbose=2,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

# Get your crew to work!
result = crew.kickoff(inputs={"company": "Google"})

print("######################")
print(result)
