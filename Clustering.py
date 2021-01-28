from SetUp import *
from DateAndTime import DateAndTime

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


class Clustering(DateAndTime):

    def __init__(self, udl, DT, folder2):

        # super(Clustering, self).__init__()
        self.udl = udl
        self.folder2 = folder2
        self.DT = DT
        self.list_cols = ['tscale', 'interest_aggressivity', 'interest_vega', 'moneyness', 'T']
        self.factor = {'tscale': 10, 'interest_aggressivity': 1, 'interest_side': 0.2, 'interest_vega': 1, 'moneyness': 0.2, 'T': 2}
        self.df_trades = pd.read_pickle(self.folder2 + '/FleshedTrades_' + self.udl + '.pkl')
        self.cluster_ratio = 3
        self.nPCA = 2

    def prepare_data(self, with_graph=False):
        self.dft = self.df_trades.copy()

        tsini = self.dft.index[0]
        self.dft['tscale'] = self.dft.apply(lambda x: self.DT.time_between(tsini, x.name), axis=1)

        self.dft.reset_index(inplace=True)
        self.dft = self.dft.loc[self.dft.pricable]

        self.dft['interest_aggressivity'] = abs(self.dft.aggressivity)


        # add trading mode (active or passive)
        self.dft['interest_side'] = self.dft.apply(lambda x: -1 if ((x.side==1) and (x.aggressivity>0)) or ((x.side==2) and (x.aggressivity<=0)) else 1, axis=1) #1 hit, -1 get hit
        self.dft['interest_vega'] = self.dft.apply(lambda x: math.copysign(1, x.aggressivity)*x.vega, axis=1)

        #center and rescale
        for elt in self.list_cols:
            self.dft[elt] = self.dft[elt] - self.dft[elt].mean()
            self.dft[elt] = self.dft[elt] / self.dft[elt].std() * self.factor[elt]

        print(self.dft[self.list_cols].head(5))
        X_normalized = np.float64(self.dft[self.list_cols].values)
        pca = PCA(n_components=self.nPCA)
        X_principal = pca.fit_transform(X_normalized)
        X_principal = pd.DataFrame(X_principal)
        X_principal.columns = ['P'+str(i) for i in range(self.nPCA)]

        if with_graph:
            plt.figure(figsize=(8, 8))
            plt.title('Visualising the data')
            Dendrogram = shc.dendrogram((shc.linkage(X_principal, method='ward')))
            plt.show()

        self.ac = AgglomerativeClustering(compute_distances=True) #n_clusters=5)
        self.ac.fit(X_principal)

        self.nodes_dic = dict(enumerate(self.ac.children_, self.ac.n_leaves_))
        self.N = X_principal.shape[0]

        self.clusters = []
        self.get_clusters(list(self.nodes_dic.keys())[-1])
        self.analyse_clusters()

        # add single leaves
        in_cluster = [elt for sublist in self.clusters for elt in sublist]
        idx = [elt for elt in range(self.dft.shape[0]) if elt not in in_cluster]
        idx_monolist = [[elt] for elt in idx]
        self.df_leaves = self.dft.iloc[idx, :]
        self.df_leaves['compo'] = idx_monolist
        self.df_leaves['timespan'] = 0
        self.df_clusters = self.df_clusters.append(self.df_leaves[self.df_clusters.columns])

        self.df_clusters['vega_intensity_abs'] = abs(self.df_clusters.vega_intensity)
        self.df_clusters['vega_intensity_per_day'] = self.df_clusters.vega_intensity_abs / np.maximum(1,self.df_clusters.timespan)
        self.df_clusters.sort_values(by='vega_intensity_per_day', ascending=False, inplace=True)
        self.df_clusters.reset_index(inplace=True)

        #compo gives the row number in dft. Index gives the row number in df_trades
        self.df_clusters['compo_in_df_trades'] = self.df_clusters['compo'].apply(lambda x: self.dft.iloc[x, :].index.tolist())

    def display_clusters(self, n=3):
        print(self.df_clusters[['timespan', 'vega_intensity', 'delta_intensity']].head(n))
        print('')
        disp_cols = ['time', 'matu', 'qty', 'PutOrCall', 'StrikePrice', 'side', 'px', 'bid', 'ask', 'aggressivity', 'vega_intensity', 'interest_aggressivity', 'interest_side', 'interest_vega']

        for i in range(n):
            print(self.dft.loc[self.df_clusters.iloc[i, :]['compo_in_df_trades'], :][disp_cols])
            print('')

    def trades(self, n):
        return self.df_clusters['compo_in_df_trades'].iloc[n]


    def get_clusters(self, node):
        s = self.nodes_dic[node]
        if s[0] >= self.N: #leaf
            a = self.get_clusters(s[0])
        else:
            a = [s[0]]

        if s[1] >= self.N:  # leaf
            b = self.get_clusters(s[1])
        else:
            b = [s[1]]

        if (s[0] > self.N) and (self.ac.distances_[node - self.N] > self.cluster_ratio * self.ac.distances_[s[0] - self.N]):
            self.clusters = self.clusters + [a]

        elif (s[1] > self.N) and (self.ac.distances_[node - self.N] > self.cluster_ratio * self.ac.distances_[s[1] - self.N]):
            self.clusters = self.clusters + [b]

        return a+b


    def analyse_clusters(self):
        self.df_clusters = pd.DataFrame()
        self.df_clusters['compo'] = self.clusters
        self.df_clusters['timespan'] = self.df_clusters.compo.apply(lambda x: self.time_between(self.dft.iloc[min(x)]['time'], self.dft.iloc[max(x)]['time'])*252) # in days with night being worth 4 hours
        self.df_clusters['vega_intensity'] = self.df_clusters.compo.apply(lambda x: self.dft.iloc[x, :]['vega_intensity'].sum())
        self.df_clusters['delta_intensity'] = self.df_clusters.compo.apply(lambda x: self.dft.iloc[x, :]['delta_intensity'].sum())


if __name__ == '__main__':
    udl = 'DBK'
    C = Clustering(udl)
    C.prepare_data()
    C.display_clusters(5)
    print(C.trades(1))
