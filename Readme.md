# ADA Project Milestone 2

## Title: Wikispeedi-Know - Hubs, User Patterns, and How They Show a Common Knowledge Gap

https://madhueb.github.io/website/


### Abstract 
Wikipedia, the largest reference work ever created, is an online encyclopedia where anyone can contribute to collaboratively written articles. Its interconnected structure makes it an ideal platform to study navigation and human associations through games like Wikispeedia (play here). In this game, players navigate from one article to another using only hyperlinks, aiming to minimize the number of clicks. This project explores two central themes: the role of hubs—highly connected articles—and user navigation patterns. By analyzing the characteristics and usage of hubs, common and abandoned paths, and more, we provide insights into how Wikipedia’s structure influences player strategies and efficiency. Our findings aim to learn something about common knowledge, user patterns, and the interplay between structure and strategy.


### Research Questions <br>
We have split our research questions into 3 key areas: <br>
1. #### Hubs <br>
* Which articles serve as the largest hubs on Wikispeedia, and how frequently are they used by players? <br>
* How well do hubs facilitate movement through the network—do they lead to faster, more efficient paths to target articles? <br> 
* What types of information are most often contained within hubs? <br> 

2. #### User Navigation Patterns <br>
* What paths are most commonly taken by players, and what does this reveal about public knowledge and associations? <br>
* Where do players tend to "block" or abandon their paths, and what kinds of topics or articles are associated with these unfinished paths? <br>
* How do common paths differ from lesser-used or “unintuitive” paths between the same articles? <br>

3. #### Comparing Hubs and User Navigation Patterns
* How do hubs affect the efficiency of user paths? Do users who navigate through hubs tend to complete paths more successfully or with fewer clicks? <br>
* Are highly linked hubs also more accessible, or do users tend to bypass them for other, potentially less-linked but more intuitive articles? <br> 
### Additional dataset (if any)
We are not planning on using any additional datasets. 

### Methods and Timeline
#### Task 1: Data Preparation and Cleaning 
**02.11.2024 - 15.11.2024** <br>
  a. Data Collection and Cleaning <br>
  We start by loading the Wikispeedia dataset, which includes user navigation paths between articles. Initial cleaning will involve removing any redundant or incomplete entries and standardizing formats for consistent data representation.

  b. Link and Article Classification <br>
  Next, we’ll take the article by topic (e.g., science, culture, geography) and categorize links by type. This will help us analyze the relationship between article topics and user navigation patterns.

#### Task 2: Hub Analysis and User Pathway Metrics
**16.11.2024 - 24.11.2024** <br>
  a: Hub Identification using PageRank Algorithm <br>
  We identify hub articles by calculating the number of links each article has, both inbound and outbound. We’ll use a PageRank algorithm to rank articles based on their connectedness and determine the largest and most influential hubs in the network.

  b: Measuring Path Efficiency with Dijkstra’s Algorithm <br>
  To evaluate how effectively hubs reduce travel distance in the network, we’ll compute shortest paths between articles using Dijkstra’s algorithm. By comparing these paths to actual user paths, we can assess the role of hubs in navigation efficiency.

#### Task 3: Comparative Analysis of Hubs and User Navigation
**25.11.2024 - 05.12.2024** <br>
  a. Analyzing Successful and Unfinished Paths <br>
  To identify where players abandon paths, we’ll categorize common "reasons" for unfinished paths. We’ll compare the unfinished tasks with successful paths and hubs to determine if certain types of paths are commonly associated with abandonment. 

  b. Path Optimization and Intuition <br>
  We’ll compare user-selected paths with optimized shortest paths, analyzing any tendency for users to bypass large hubs in favor of more intuitive but less connected links. This will reveal patterns in user decision-making.

#### Task 4: Reporting Insights and Answering Research Questions, Final Report Compilation
**06.12.202 - 18.12.2024** <br>
  a. Interpret Findings on Knowledge Gaps and Cultural Bias <br>
  Finally, we’ll analyze the results to draw conclusions about public knowledge, intuitive navigation, and cultural biases, using our findings on common vs. specialized      knowledge, navigation patterns, and hub influence.

### Organization within the team 
* Michelle:
  * complete PageRank Algorithm Analysis by 24.11.24
  * write report and analysis conclusions on knowledge gaps by 15.12.24 
* Viktor: conduct preliminary research on shortest paths and link network
  * complete path optimization and intuition by 5.12.24
* Antoine: conduct preliminary research on unfinished paths despite small semantic distance
  * complete analysis on successful and unfinished paths by 5.12.24
  * create webpage to host results on by 5.12.24   
* Lisa: conduct preliminary research on dataset and worked on reading in the data.
  * measure path efficiency with Dijkstra's algorithm by 24.11.24
  * identify hub correlation to common knowledge by 15.12.24
* Madeleine : conduct preliminary research on behaviour similarity 
  * analyze game similarities between successful and unsuccesful ones by 5.12.24
  * compile final report by 15.12.24
### Questions for TA

None
