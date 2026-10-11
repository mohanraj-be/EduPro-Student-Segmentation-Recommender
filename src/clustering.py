"""
clustering.py - Unsupervised learner segmentation, validation, dimensionality reduction, and persona profiling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class LearnerSegmenter:
    """
    Executes unsupervised clustering algorithms (K-Means, Hierarchical),
    calculates validation metrics, generates 2D/3D projections (PCA, t-SNE),
    and creates interpretable learner archetype profiles.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans_model: KMeans = None
        self.hierarchical_model: AgglomerativeClustering = None
        self.pca_2d: PCA = None
        self.pca_3d: PCA = None
        self.tsne_2d: TSNE = None
        self.feature_names: List[str] = []
        self.cluster_names_map: Dict[int, str] = {}
        self.cluster_descriptions: Dict[int, str] = {}

    def fit_transform(self, X_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Scales features, fits K-Means and Hierarchical clustering models,
        and generates dimensionality reduction embeddings.
        """
        self.feature_names = X_df.columns.tolist()
        X_scaled = self.scaler.fit_transform(X_df)

        # Train primary K-Means model
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=15,
            max_iter=400,
            random_state=self.random_state
        )
        kmeans_labels = self.kmeans_model.fit_predict(X_scaled)

        # Train validation Hierarchical model
        self.hierarchical_model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage='ward'
        )
        hierarchical_labels = self.hierarchical_model.fit_predict(X_scaled)

        # Fit PCA 2D & 3D
        self.pca_2d = PCA(n_components=2, random_state=self.random_state)
        pca_2d_coords = self.pca_2d.fit_transform(X_scaled)

        self.pca_3d = PCA(n_components=3, random_state=self.random_state)
        pca_3d_coords = self.pca_3d.fit_transform(X_scaled)

        # Build results DataFrame
        result_df = X_df.copy()
        result_df['cluster'] = kmeans_labels
        result_df['hierarchical_cluster'] = hierarchical_labels
        result_df['pca_x'] = pca_2d_coords[:, 0]
        result_df['pca_y'] = pca_2d_coords[:, 1]
        result_df['pca_z'] = pca_3d_coords[:, 2]

        # Formulate cluster persona archetype names
        self._generate_cluster_personas(result_df)
        result_df['cluster_name'] = result_df['cluster'].map(self.cluster_names_map)

        return X_scaled, result_df

    def compute_cluster_metrics_sweep(self, X_df: pd.DataFrame, k_range: range = range(2, 9)) -> Dict[str, List[Any]]:
        """
        Computes clustering validation metrics across varying cluster counts (Elbow & Silhouette).
        """
        X_scaled = self.scaler.fit_transform(X_df)
        
        metrics = {
            'k_values': list(k_range),
            'inertia': [],
            'silhouette_scores': [],
            'davies_bouldin_scores': [],
            'calinski_harabasz_scores': [],
            'hierarchical_silhouette_scores': []
        }

        for k in k_range:
            # K-Means
            km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=self.random_state)
            km_labels = km.fit_predict(X_scaled)
            metrics['inertia'].append(float(km.inertia_))
            metrics['silhouette_scores'].append(float(silhouette_score(X_scaled, km_labels)))
            metrics['davies_bouldin_scores'].append(float(davies_bouldin_score(X_scaled, km_labels)))
            metrics['calinski_harabasz_scores'].append(float(calinski_harabasz_score(X_scaled, km_labels)))

            # Hierarchical
            hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
            hc_labels = hc.fit_predict(X_scaled)
            metrics['hierarchical_silhouette_scores'].append(float(silhouette_score(X_scaled, hc_labels)))

        return metrics

    def compute_tsne_projection(self, X_scaled: np.ndarray, perplexity: int = 30) -> np.ndarray:
        """
        Generates 2D t-SNE projection for non-linear manifold visual exploration.
        """
        self.tsne_2d = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=self.random_state,
            max_iter=1000
        )
        return self.tsne_2d.fit_transform(X_scaled)

    def _generate_cluster_personas(self, segmented_df: pd.DataFrame):
        """
        Analyzes feature centroid profiles to assign crisp, mathematically grounded persona labels.
        """
        means = segmented_df.groupby('cluster').mean(numeric_only=True)
        
        for c_id in range(self.n_clusters):
            c_mean = means.loc[c_id]
            total_courses = c_mean.get('total_courses_enrolled', 0)
            diversity = c_mean.get('diversity_score', 0)
            depth = c_mean.get('learning_depth_index', 0.5)
            adv_ratio = c_mean.get('level_ratio_Advanced', 0)
            beg_ratio = c_mean.get('level_ratio_Beginner', 0)
            int_ratio = c_mean.get('level_ratio_Intermediate', 0)

            if total_courses >= 8 or diversity >= 0.5:
                name = "High-Engagement Super-Learners & Multidisciplinary Achievers"
                desc = "Power users with broad horizontal interests across multiple domains, high learning hours, and significant platform investment."
            elif adv_ratio >= 0.7 or depth >= 0.85:
                name = "Deep Domain Specialists & Advanced Experts"
                desc = "Direct-to-mastery professionals focusing exclusively on high-level specialized topics and technical competence."
            elif beg_ratio >= 0.7 or depth <= 0.45:
                name = "Foundational & Entry-Level Starters"
                desc = "Exploratory learners starting with introductory curricula, building fundamental concepts and baseline skills."
            elif int_ratio >= 0.4 or (0.50 <= depth <= 0.75):
                name = "Intermediate Applied Practitioners"
                desc = "Balanced learners actively bridging fundamentals with applied intermediate topics and industry practical workflows."
            else:
                name = f"Learner Persona Segment {c_id + 1}"
                desc = "Engaged learners with balanced multi-level course progression."

            self.cluster_names_map[c_id] = name
            self.cluster_descriptions[c_id] = desc

    def predict_new_learner_cluster(self, new_features_df: pd.DataFrame) -> Tuple[int, str]:
        """
        Assigns cluster and persona to an ad-hoc or incoming learner profile.
        """
        # Ensure column order
        aligned_df = new_features_df[self.feature_names].copy()
        X_new_scaled = self.scaler.transform(aligned_df)
        cluster_id = int(self.kmeans_model.predict(X_new_scaled)[0])
        cluster_name = self.cluster_names_map.get(cluster_id, f"Segment {cluster_id}")
        return cluster_id, cluster_name

    def get_cluster_summary_table(self, segmented_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates clean centroid summary comparison across all clusters.
        """
        summary_cols = [
            'total_courses_enrolled', 'total_spend', 'avg_spend_per_course',
            'learning_depth_index', 'diversity_score', 'specialization_ratio',
            'paid_course_ratio', 'total_learning_hours', 'avg_course_rating_enrolled', 'Age'
        ]
        valid_cols = [c for c in summary_cols if c in segmented_df.columns]
        summary = segmented_df.groupby('cluster')[valid_cols].mean().round(2)
        summary['Learner_Count'] = segmented_df.groupby('cluster').size()
        summary['Percentage'] = ((summary['Learner_Count'] / len(segmented_df)) * 100).round(1).astype(str) + '%'
        summary['Segment_Persona'] = [self.cluster_names_map.get(i, f"Segment {i}") for i in summary.index]
        return summary
