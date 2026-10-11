"""
run_pipeline.py - Orchestration script for full end-to-end data processing, clustering, recommendation evaluation, and artifact serialization.
"""

import os
import json
import pandas as pd
import numpy as np
from src.data_loader import EduProDataLoader
from src.feature_engineering import LearnerFeatureEngineer
from src.clustering import LearnerSegmenter
from src.recommender import ClusterAwareRecommender
from src.evaluation import EduProEvaluator


def main():
    print("=" * 70)
    print(" EduPro Student Segmentation & Personalized Recommender Pipeline")
    print("=" * 70)

    # 1. Ingest Data
    print("\n[1/6] Ingesting EduPro platform dataset...")
    data_loader = EduProDataLoader(excel_path="EduPro Online Platform.xlsx")
    data_dict = data_loader.load_data()
    summary_stats = data_loader.get_summary_stats()
    print(f" Loaded: {summary_stats['total_users']} Users, {summary_stats['total_courses']} Courses, {summary_stats['total_transactions']} Transactions, {summary_stats['total_teachers']} Teachers")

    # 2. Feature Engineering
    print("\n[2/6] Engineering learner-level engagement, preference, and behavioral features...")
    feature_engineer = LearnerFeatureEngineer(data_dict)
    learners_df = feature_engineer.construct_learner_features()
    X_matrix, feature_cols = feature_engineer.get_feature_matrix_for_clustering(learners_df)
    print(f" Extracted {len(feature_cols)} clustering features for {len(learners_df)} learners.")
    print(f" Features: {', '.join(feature_cols[:8])} ...")

    # 3. Model Selection Sweep & Learner Segmentation
    print("\n[3/6] Performing optimal cluster selection sweep (K=2 to 8)...")
    segmenter = LearnerSegmenter(n_clusters=4, random_state=42)
    k_sweep = segmenter.compute_cluster_metrics_sweep(X_matrix, k_range=range(2, 9))
    print(f" Silhouette scores across K: {dict(zip(k_sweep['k_values'], [round(s, 4) for s in k_sweep['silhouette_scores']]))}")

    print("\n[4/6] Fitting optimal K-Means & Hierarchical models (K=4), PCA 2D/3D & t-SNE...")
    X_scaled, segmented_df = segmenter.fit_transform(X_matrix)
    
    # Add metadata back
    for meta_col in ['UserID', 'UserName', 'Age', 'Gender', 'Email', 'primary_category', 'primary_course_level', 'preferred_course_type']:
        if meta_col in learners_df.columns:
            segmented_df[meta_col] = learners_df[meta_col].values

    # Compute t-SNE coordinates for visualization
    tsne_coords = segmenter.compute_tsne_projection(X_scaled, perplexity=30)
    segmented_df['tsne_x'] = tsne_coords[:, 0]
    segmented_df['tsne_y'] = tsne_coords[:, 1]

    # Cluster Summary
    cluster_summary = segmenter.get_cluster_summary_table(segmented_df)
    print("\n Learner Archetype Personas Identified:")
    for c_id, name in segmenter.cluster_names_map.items():
        count = (segmented_df['cluster'] == c_id).sum()
        pct = (count / len(segmented_df)) * 100
        print(f"  • Cluster {c_id}: {name} -> {count} learners ({pct:.1f}%) | {segmenter.cluster_descriptions[c_id]}")

    # 4. Initialize Cluster-Aware Recommender
    print("\n[5/6] Initializing Cluster-Aware Hybrid Recommender Engine...")
    recommender = ClusterAwareRecommender(
        courses_df=data_dict['courses'],
        transactions_df=data_dict['transactions'],
        learners_df=segmented_df
    )
    
    # Test sample recommendation
    sample_user = segmented_df['UserID'].iloc[0]
    sample_recs = recommender.get_recommendations_for_learner(sample_user, top_n=3)
    print(f"\n Sample Recommendations for Learner {sample_user} ({segmented_df['cluster_name'].iloc[0]}):")
    for _, r in sample_recs.iterrows():
        print(f"  -> [{r['RecommendationScore']}% Match] {r['CourseName']} ({r['CourseLevel']} | {r['CourseCategory']}) - {r['Reasoning']}")

    # 5. Comprehensive Evaluation
    print("\n[6/6] Computing evaluation metrics & engagement lift estimations...")
    evaluator = EduProEvaluator(
        segmented_learners_df=segmented_df,
        feature_matrix_scaled=X_scaled,
        courses_df=data_dict['courses'],
        transactions_df=data_dict['transactions']
    )
    
    cluster_eval = evaluator.evaluate_clustering_quality()
    cohesion_eval = evaluator.evaluate_intra_cluster_cohesion()
    rec_eval = evaluator.evaluate_recommendation_metrics(recommender, test_sample_size=300, top_k=5)
    engagement_lift = evaluator.estimate_engagement_lift()

    all_metrics = {
        'clustering_quality': cluster_eval,
        'intra_cluster_cohesion': cohesion_eval,
        'recommendation_metrics': rec_eval,
        'engagement_lift_proxy': engagement_lift,
        'cluster_archetypes': {
            str(k): {'name': segmenter.cluster_names_map[k], 'description': segmenter.cluster_descriptions[k]}
            for k in segmenter.cluster_names_map
        }
    }

    # 6. Save Artifacts
    os.makedirs("data", exist_ok=True)
    segmented_df.to_csv("data/processed_learners.csv", index=False)
    cluster_summary.to_csv("data/cluster_profiles.csv")
    data_dict['courses'].to_csv("data/courses_processed.csv", index=False)
    
    with open("data/evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open("data/k_sweep_metrics.json", "w") as f:
        json.dump(k_sweep, f, indent=2)

    print("\n" + "=" * 70)
    print(" Pipeline executed successfully! All artifacts exported to data/")
    print(f"  - data/processed_learners.csv ({len(segmented_df)} records)")
    print(f"  - data/cluster_profiles.csv")
    print(f"  - data/courses_processed.csv")
    print(f"  - data/evaluation_metrics.json")
    print(f"  - data/k_sweep_metrics.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
