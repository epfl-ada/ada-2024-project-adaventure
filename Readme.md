# Wikispeedi-Know - Hubs, User Patterns, and How They Show a Common Knowledge Gap
You can explore the data story and its findings by visiting this [link](https://madhueb.github.io/website/) to the data story. 

### Abstract
Wikipedia, the largest reference work ever created, is an online encyclopedia where anyone can contribute to collaboratively written articles. Its interconnected structure makes it an ideal platform to study navigation and human associations through games like Wikispeedia (play here). In this game, players navigate from one article to another using only hyperlinks, aiming to minimize the number of clicks. This project explores two central themes: the role of hubs—highly connected articles—and user navigation patterns. By analyzing the characteristics and usage of hubs, common and abandoned paths, and more, we provide insights into how Wikipedia’s structure influences player strategies and efficiency. Our findings aim to learn something about common knowledge, user patterns, and the interplay between structure and strategy.

### Research Questions <br>
We have split our research questions into 3 key areas: <br>
1. #### Hubs <br>
* What are the defining characteristics of a hub?
* Which articles serve as the largest hubs on Wikispeedia, and how frequently are they used by players?
* What types of information are most often contained within hubs, and how are the hubs related to other articles or subject categories?

2. #### User Navigation Patterns <br>
* What paths are most commonly taken by players, and what does this reveal about public knowledge and associations?
* Where do players tend to "block" or abandon their paths, and what kinds of topics or articles are associated with these unfinished paths?
* Do users exhibit similar behaviors when navigating Wikipedia’s network?

3. #### Comparing Hubs and User Navigation Patterns
* To what extent does the reliance on hubs obscure the identification of common knowledge?
* Is it possible to identify common knowledge in a navigational context despite the influence of network structure?

### File Structure
**`analysis.ipynb/`**: The main code for analysis contained in the data story is located here. This Jupyter notebook includes the primary logic and calculations for exploring Wikipedia navigation patterns and analyzing hubs.

**`src/`**: Contains additional Python scripts that support the analysis. This folder includes data processing scripts and other components necessary for running the full analysis.

**`data/`**: Stores the data used throughout the project.

**`images/`**: Contains images for the data story, organized into two subfolders:
- **`html/`**: Includes HTML figures generated for the project.
- **`png/`**: Includes PNG figures generated for the project.

### External data
* In order to validate the results in part 3 an external dataset of monthly visits to Wikipedia articles was gathered using the Wikimedia api.

### Organization within the team 
* Michelle:
  * Wrote algorithm for PageRank and analysed results
  * Worked on hubs data story and Introduction
* Viktor: 
  * Created User Frequency vs PageRank score plots and analysed results
  * Worked on the data story 
* Antoine: 
  * Analysed Forks in user navigation paths
  * Worked on the data story
* Lisa: 
  * Worked on introduction and introduction to wikipedia
  * Worked on analysis of PageRank score and hubs data story 
* Madeleine : 
  * Analysed user navigation patterns 
  * Worked on data story
 
  
 
### Original Research Questions
The final research questions for part 3 were slightly altered from the original ones. The reason being that further analysis showed the original ones asked questions that were a bit too difficult to answer with the given data. The original research questions for part 3 can be seen below.

3. #### Comparing Hubs and User Navigation Patterns
* Can we determine areas of common knowledge or knowledge gaps by comparing user patterns with hubs?
* How do hubs affect the efficiency of user paths? Do users who navigate through hubs tend to complete paths more successfully or with fewer clicks?
* Are highly linked hubs also more accessible, or do users tend to bypass them for other, potentially less-linked but more intuitive articles?

### Citations

* Nils Reimers, Iryna Gurevych. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. arXiv. https://arxiv.org/abs/ArticleID
* L. Page, S. Brin, R. Motwani, and T. Winograd. (1999). The PageRank Citation Ranking: Bringing Order to the Web.  Stanford InfoLab. http://ilpubs.stanford.edu:8090/422/

