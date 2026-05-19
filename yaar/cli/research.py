from .session import run_agent

def main():
    run_agent('commands/research_codebase.md', agent_name='Reserch_agent', description='Code research agent') 

if __name__ == "__main__":
    main()
