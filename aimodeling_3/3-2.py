import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width', color='species') # petal_length를 제외하여 3차원 공간 구성
fig.show(renderer="browser")