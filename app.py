from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # ✅ Necessário para usar matplotlib no Flask sem GUI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

app = Flask(__name__)

# ========= 1. CARREGAR E PREPARAR DADOS =========
df = pd.read_excel('dados_ibge.xlsx', header=None, skiprows=6, nrows=3)

df.columns = ['Padrao', 'Residencial_3q', 'Residencial_4q']
df['Padrao_num'] = [0, 1, 2]  # Ajuste conforme a ordem dos seus dados

df_long = pd.melt(df, id_vars=['Padrao', 'Padrao_num'],
                  value_vars=['Residencial_3q', 'Residencial_4q'],
                  var_name='Tipo', value_name='Custo_m2')
df_long['Tipo_num'] = df_long['Tipo'].map({'Residencial_3q': 0, 'Residencial_4q': 1})

# Dados para clustering
X = df_long[['Tipo_num', 'Padrao_num', 'Custo_m2']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========= 2. TREINAMENTO DOS MODELOS =========
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
meanshift = MeanShift().fit(X_scaled)

# ========= 3. GERAR GRÁFICOS =========
def gerar_graficos():
    # --- Gráfico 2D com PCA ---
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=80)
    plt.title('Clusters - Visualização 2D (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.savefig('static/grafico_2d.png')
    plt.close()

    # --- Gráfico 3D ---
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
               c=kmeans.labels_, cmap='viridis', s=80)
    ax.set_title('Clusters - Visualização 3D')
    ax.set_xlabel('Tipo')
    ax.set_ylabel('Padrão')
    ax.set_zlabel('Custo/m²')
    plt.tight_layout()
    plt.savefig('static/grafico_3d.png')
    plt.close()

# Gera os gráficos ao iniciar a aplicação
if not os.path.exists('static'):
    os.makedirs('static')
gerar_graficos()

# ========= 4. ROTAS FLASK =========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    metragem = float(request.form['metragem'])
    tipo = int(request.form['tipo'])       # 0 ou 1
    padrao = int(request.form['padrao'])   # 0, 1 ou 2

    # Obtem custo médio compatível com os inputs
    filtro = (df_long['Tipo_num'] == tipo) & (df_long['Padrao_num'] == padrao)
    custo_m2 = df_long.loc[filtro, 'Custo_m2'].mean()
    custo_total = metragem * custo_m2

    return render_template('resultado.html',
                           metragem=metragem,
                           custo_m2=custo_m2,
                           custo_total=custo_total,
                           grafico2d='static/grafico_2d.png',
                           grafico3d='static/grafico_3d.png')

if __name__ == '__main__':
    app.run(debug=True)
