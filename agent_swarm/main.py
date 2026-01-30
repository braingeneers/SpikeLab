import argparse
import os
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
import logging
from agent_swarm.core.swarm import SwarmOrchestrator

# Suppress AutoGen and OpenAI API key format warnings
# logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)
# logging.getLogger("openai").setLevel(logging.ERROR)
<<<<<<< HEAD
=======
from agent_swarm.core.swarm import SwarmOrchestrator

>>>>>>> ba3f8a9 (initial version)
=======
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
=======
from agent_swarm.core.swarm import SwarmOrchestrator

>>>>>>> ba3f8a9 (initial version)

def main():
    parser = argparse.ArgumentParser(description="Agent Swarm Orchestrator")
    parser.add_argument(
        "objective",
        type=str,
        nargs="?",
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        help="The goal for the swarm to achieve. If omitted, the swarm listens on Slack. 4. Signal TERMINATE_SWARM when complete.\n",
=======
        help="The goal for the swarm to achieve. If omitted, the swarm listens on Slack.",
>>>>>>> ba3f8a9 (initial version)
=======
        help="The goal for the swarm to achieve. If omitted, the swarm listens on Slack. 4. Signal TERMINATE_SWARM when complete.\n",
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
=======
        help="The goal for the swarm to achieve. If omitted, the swarm listens on Slack.",
>>>>>>> ba3f8a9 (initial version)
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    orchestrator = SwarmOrchestrator(objective=args.objective)
    orchestrator.run()


if __name__ == "__main__":
    main()
