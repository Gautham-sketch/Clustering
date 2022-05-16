from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import csv

rows = []
with open("Final_Project.csv",'r') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)

headers = rows[0]
star_data = rows[1: ]
star_mass = star_data[6]
star_radius = star_data[5]

Y = []
for index,mass in enumerate(star_mass):
  temp = [star_radius[index],mass]
  Y.append(temp)
print(Y)

wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i,init="k-means++",random_state=42)
  kmeans.fit(Y)
  wcss.append(kmeans.inertia_)

plt.Figure(figsize = (10,5))
sns.lineplot(range(1,11),wcss,marker = 'o',color = 'red')
plt.title("Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()