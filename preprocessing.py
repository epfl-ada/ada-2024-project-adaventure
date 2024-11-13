import pandas as pd
import numpy as np
from urllib.parse import unquote

class WikispeediaData:
    def __init__(self):
        self.articles = pd.read_csv('articles.tsv', sep='\t',comment='#', header = None, encoding="utf-8", names = [ 'article_name'])
        self.categories = pd.read_csv('categories.tsv', sep='\t',comment='#', header = None, encoding="utf-8", names = ['article_name', 'category_name'])
        self.links = pd.read_csv('links.tsv', sep='\t',comment='#', header = None, encoding="utf-8", names = ['1st article', '2nd article'])
        self.paths_finished = pd.read_csv('paths_finished.tsv', sep='\t',comment='#', header = None, encoding="utf-8", names = ['hashedIpAddress',  'timestamp',   'durationInSec',   'path',   'rating'])
        self.paths_unfinished = pd.read_csv('paths_unfinished.tsv', sep='\t',comment='#', header = None, encoding="utf-8", names = ['hashedIpAddress',  'timestamp',   'durationInSec',   'path','target','rating'])
        self.matrix = self.load_shortest_paths()

        self.preprocess()



    def load_shortest_paths(self):
        file_path = 'shortest-path-distance-matrix.txt'

        # Read file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Ignore comment lines
        data_lines = [line.strip() for line in lines if not line.startswith('#')]

        matrix = []

        # Replace '_' with np.inf and convert numbers to integers
        for line in data_lines[1:]:
            row = []
            for char in line:
                if char == '_':
                    row.append(np.inf)  # Inaccessible paths
                else:
                    try:
                        row.append(int(char))  # Convert to integer
                    except ValueError:
                        continue  # Skip invalid characters
            matrix.append(row)

        # Convert to numpy array
        matrix_np = np.array(matrix)
        return matrix_np

    def get_category(self,categorie):
        l = categorie.split('.')
        return l[1:]
    
    def different_cat(self,link):
        article_1 = link["1st article"]
        article_2 = link["2nd article"]
        category_1 = self.categories["1st cat"][self.categories["article_name"]== article_1]
        category_2 = self.categories["1st cat"][self.categories["article_name"]== article_2]
        if category_1.empty or category_2.empty :
            return None
        if category_1.values[0]==category_2.values[0] :
            return 0
        else : 
            return 1
    
    def back_clicks_2(self,path):
        for i in range(len(path)-1):
            if path[i] == '<':
                path[i] = path[i-2] 
        return path

    def back_clicks(self,path):
        while '<' in path:
            i = path.index('<')
            del path[i]
            del path[i-1]
        return path
    

    def preprocess(self):

        #Preprocess articles
        self.articles['article_name'] = self.articles['article_name'].apply(lambda x: unquote(x).replace('_', ' '))
    
        #Preprocess categories
        self.categories['article_name'] = self.categories['article_name'].apply(lambda x: unquote(x).replace('_', ' '))
        self.categories['category_name'] = self.categories['category_name'].apply(lambda x: self.get_category(x))
        self.categories["1st cat"] = self.categories['category_name'].apply(lambda x: unquote(x[0]).replace('_',' ') if len(x) > 0 else None)
        self.categories["2nd cat"] = self.categories['category_name'].apply(lambda x: unquote(x[1]).replace('_',' ') if len(x) > 1 else None)
        self.categories["3rd cat"] = self.categories['category_name'].apply(lambda x: unquote(x[2]).replace('_',' ') if len(x) > 2 else None)

        self.categories = self.categories.drop('category_name', axis=1)

        #Preprocess links
        self.links['1st article'] = self.links['1st article'].apply(lambda x: unquote(x).replace('_', ' '))
        self.links['2nd article'] = self.links['2nd article'].apply(lambda x: unquote(x).replace('_', ' '))
        self.links['different_cat'] = self.links.apply(lambda x: self.different_cat(x), axis=1)

        #Preprocess paths_finished

        self.paths_finished["path"] = self.paths_finished["path"].apply(lambda x: unquote(x).replace('_', ' '))
        self.paths_finished["path"]= self.paths_finished["path"].apply(lambda x: x.split(';'))
        self.paths_finished["paths"] = self.paths_finished["path"].apply(lambda x: self.back_clicks(x))

        #Preprocess paths_unfinished

        self.paths_unfinished["path"] = self.paths_unfinished["path"].apply(lambda x: unquote(x).replace('_', ' '))
        self.paths_unfinished["target"] = self.paths_unfinished["target"].apply(lambda x: unquote(x).replace('_', ' '))
        self.paths_unfinished["path"]= self.paths_unfinished["path"].apply(lambda x: x.split(';'))
        self.paths_unfinished["paths"] = self.paths_unfinished["path"].apply(lambda x: self.back_clicks(x))

        #Count articles in paths    
        self.count_articles()

    def count_articles(self):
        self.articles ["nb_in_unfinished_paths"] = self.articles["article_name"].apply(lambda x: self.paths_unfinished["path"].str.contains(x,regex=False).sum())
        self.articles ["nb_in_finished_paths"] = self.articles["article_name"].apply(lambda x: self.paths_finished["path"].str.contains(x,regex=False).sum())
        self.articles["nb_links"]=self.articles["article_name"].apply(lambda x: self.links["1st article"].str.contains(x,regex=False).sum()+self.links["2nd article"].str.contains(x,regex=False).sum())
        #self.articles["nb_links_same_cat"]=self.articles["article_name"].apply(lambda x: self.links[self.links["1st article"]==x]["different_cat"].sum()+self.links[self.links["2nd article"]==x]["different_cat"].sum())


    def get_nb(self,categorie,category_type = "1st cat",path = "nb_in_finished_paths"):
        s=0
        for i in range(len(self.categories)):
            if self.categories[category_type][i] == categorie:
                article_name = self.categories["article_name"][i]
                s += self.articles[path][self.articles["article_name"]==article_name].values[0]
        return s

    def count_cat(self):
        cat_1 = self.categories.groupby("1st cat").size().reset_index(name='counts')
        cat_2 = self.categories.groupby(["1st cat","2nd cat"]).size().reset_index(name='counts')

        cat_1["nb_in_finished_paths"] = cat_1["1st cat"].apply(lambda x: self.get_nb(x,path ="nb_in_finished_paths"))
        cat_1["nb_in_unfinished_paths"] = cat_1["1st cat"].apply(lambda x: self.get_nb(x,path ="nb_in_unfinished_paths"))
        cat_1["nb_links"] = cat_1["1st cat"].apply(lambda x: self.get_nb(x,path ="nb_links"))

        cat_2["nb_in_finished_paths"] = cat_2["2nd cat"].apply(lambda x: self.get_nb(x,"2nd cat","nb_in_finished_paths"))
        cat_2["nb_in_unfinished_paths"] = cat_2["2nd cat"].apply(lambda x: self.get_nb(x,"2nd cat","nb_in_unfinished_paths"))
        cat_2["nb_links"] = cat_2["2nd cat"].apply(lambda x: self.get_nb(x,"2nd cat","nb_links"))
        return cat_1,cat_2