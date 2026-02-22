from Bio import Entrez
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Dict, Any, List, Callable
=======
from typing import Dict, Any, List
>>>>>>> c98998a (Implement: take a look at this and implement the functional connectivity metric from this paper: <https://pubmed.ncbi.nlm.nih.gov/29024669/>)
=======
from typing import Dict, Any, List, Callable
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
import os


class PubMedTools:
    def __init__(self):
        # Always tell NCBI who you are
        Entrez.email = os.getenv("ENTREZ_EMAIL", "agent_swarm@example.com")
        Entrez.tool = "AgentSwarmResearcher"

<<<<<<< HEAD
<<<<<<< HEAD
    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
=======
    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
>>>>>>> c98998a (Implement: take a look at this and implement the functional connectivity metric from this paper: <https://pubmed.ncbi.nlm.nih.gov/29024669/>)
=======
    def search_pubmed(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
        """
        Search PubMed for papers matching the query.
        Returns a list of dictionaries with title, pmid, and summary (if available in snippet).
        """
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            handle.close()

            id_list = record["IdList"]
            if not id_list:
                return []

            # Fetch details for the found IDs
            return self.fetch_details(id_list)

        except Exception as e:
            return [{"error": str(e)}]

    def fetch_details(self, id_list: List[str]) -> List[Dict[str, str]]:
        """
        Fetch detailed information for a list of PMIDs.
        """
        try:
            ids = ",".join(id_list)
            handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            results = []
            for paper in records["PubmedArticle"]:
                try:
                    article = paper["MedlineCitation"]["Article"]
                    journal_info = paper["MedlineCitation"]["MedlineJournalInfo"]

                    title = article.get("ArticleTitle", "No title")
                    abstract_list = article.get("Abstract", {}).get("AbstractText", [])
                    abstract = (
                        " ".join(abstract_list)
                        if abstract_list
                        else "No abstract available."
                    )

                    authors_list = article.get("AuthorList", [])
                    authors = ", ".join(
                        [
                            f"{a.get('LastName', '')} {a.get('Initials', '')}"
                            for a in authors_list
                        ]
                    )

                    pmid = str(paper["MedlineCitation"]["PMID"])

                    results.append(
                        {
                            "pmid": pmid,
                            "title": title,
                            "authors": authors,
                            "journal": article.get("Journal", {}).get("Title", ""),
                            "year": article.get("Journal", {})
                            .get("JournalIssue", {})
                            .get("PubDate", {})
                            .get("Year", ""),
                            "abstract": abstract,
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        }
                    )
                except Exception as e:
                    results.append(
                        {"pmid": "Unknown", "error": f"Error parsing paper: {str(e)}"}
                    )

            return results

        except Exception as e:
            return [{"error": str(e)}]

    def get_paper_by_pmid(self, pmid: str) -> Dict[str, str]:
        """
        Fetch details for a single PMID.
        """
        results = self.fetch_details([pmid])
        return results[0] if results else {"error": "Paper not found"}

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
    def get_tool_map(self) -> Dict[str, Callable]:
        return {
            "search_pubmed": self.search_pubmed,
            "get_paper_details": self.get_paper_by_pmid,
        }

<<<<<<< HEAD
=======
>>>>>>> c98998a (Implement: take a look at this and implement the functional connectivity metric from this paper: <https://pubmed.ncbi.nlm.nih.gov/29024669/>)
=======
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)

pubmed_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": "Search PubMed for scientific papers and return their abstracts and metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search keywords or query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max number of papers to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_details",
            "description": "Get valid title, abstract, and metadata for a specific PubMed ID (PMID).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "string",
                        "description": "The PubMed ID of the paper.",
                    }
                },
                "required": ["pmid"],
            },
        },
    },
]
